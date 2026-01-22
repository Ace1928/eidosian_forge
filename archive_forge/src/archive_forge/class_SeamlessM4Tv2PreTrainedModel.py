import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SeamlessM4Tv2Config
    base_model_prefix = 'seamless_m4t_v2'
    supports_gradient_checkpointing = True
    _no_split_modules = ['SeamlessM4Tv2EncoderLayer', 'SeamlessM4Tv2DecoderLayer', 'SeamlessM4Tv2ConformerEncoderLayer', 'SeamlessM4Tv2TextToUnitDecoderLayer']

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4Tv2ConformerSelfAttention):
            if hasattr(module, 'pos_bias_u'):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, 'pos_bias_v'):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4Tv2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        kernel_size, stride = (self.config.adaptor_kernel_size, self.config.adaptor_stride)
        pad = kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
        seq_lens = (seq_lens + 2 * pad - kernel_size) / stride + 1
        return seq_lens.floor()

    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        """
        if not hasattr(self.generation_config, 'id_to_text'):
            raise ValueError("This model generation config doesn't have a `id_to_text` key which maps\n                token ids to subwords. Make sure to load the right generation config.")
        batch_size, sequence_len = input_ids.shape
        subwords_batch = []
        for batch_id in range(batch_size):
            subwords = []
            for i in range(sequence_len):
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    def _count_character_length_in_subword(self, input_ids, subwords_batch, merge_space_with_prev_subword=False, pad_token_id=0, unk_token_id=1, space='▁'):
        """
        Counts the number of characters per text string associated with the input token id.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            merge_space_with_prev_subword (`bool`, *optional*, defaults to `False`):
                Indicates if the space character is merged with the previous subword. If `False`, it will be merged
                with the next subword.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
            space (`str`, *optional*, defaults to `"▁"`):
                The space character.
        """
        batch_size, _ = input_ids.shape
        char_count_per_id = input_ids.new_zeros(input_ids.size())
        subword_lens = input_ids.ne(pad_token_id).sum(1)
        for batch_id in range(batch_size):
            subword_indices = input_ids[batch_id, :subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][:subword_lens[batch_id]]
            is_next_start_with_space = [len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < len(subwords) - 1 else False for i in range(len(subwords))]
            is_punc = [len(subwords[i]) == 1 and (not subwords[i].isalpha()) and (not subwords[i].isnumeric()) and (subwords[i] != space) for i in range(len(subwords))]
            for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
                if subword_idx == pad_token_id:
                    break
                if subword_idx == unk_token_id:
                    char_len = 1
                    if merge_space_with_prev_subword and is_next_start_with_space[i]:
                        char_len += 1
                else:
                    char_len = len(subword)
                    if merge_space_with_prev_subword:
                        if is_next_start_with_space[i]:
                            char_len += 1
                        if i > 0 and is_next_start_with_space[i - 1]:
                            char_len -= 1
                    elif is_punc[i] and is_next_start_with_space[i]:
                        char_len += 1
                    elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                        char_len -= 1
                char_count_per_id[batch_id, i] = char_len
        return char_count_per_id

    def _get_char_input_ids(self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1):
        """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `torch.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
        if not hasattr(self.generation_config, 'char_to_id'):
            raise ValueError("This model generation config doesn't have a `char_to_id` key which maps\n                characters to character ids. Make sure to load the right generation config.")
        batch_size = input_ids.shape[0]
        max_len = int(char_count_per_id.sum(1).max().item())
        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)
        subword_lens = input_ids.ne(pad_token_id).sum(1)
        for batch_id in range(batch_size):
            total = 0
            subword_indices = input_ids[batch_id, :subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][:subword_lens[batch_id]]
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    char_ids = [unk_token_id]
                else:
                    char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
                char_seq_len = len(char_ids)
                char_seqs[batch_id, total:total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
                total += char_seq_len
        return char_seqs

    def _hard_upsample(self, hidden_states, durations):
        """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
        else:
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning_once('`self.training=True` and you use batching. You lose parallelism during the hifigan\n                               forward pass because the samples are interleaved.')
            hidden_states = [torch.repeat_interleave(hidden_state, duration, dim=0) for hidden_state, duration in zip(hidden_states, durations)]
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
        return hidden_states