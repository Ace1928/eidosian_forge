import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@add_start_docstrings('Bark coarse acoustics model.\n    It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a\n    language modeling head on top.', BARK_MODEL_START_DOCSTRING.format(config='BarkCoarseConfig'))
class BarkCoarseModel(BarkCausalModel):
    base_model_prefix = 'coarse_acoustics'
    config_class = BarkCoarseConfig

    def preprocess_histories(self, max_coarse_history: int, semantic_to_coarse_ratio: int, batch_size: int, semantic_generation_config: int, codebook_size: int, history_prompt: Optional[Dict[str, torch.Tensor]]=None):
        """
        Preprocess the optional `Bark` speaker prompts before `self.generate`.

        Args:
            max_coarse_history (`int`):
                Maximum size of coarse tokens used.
            semantic_to_coarse_ratio (`int`):
                Ratio of semantic to coarse frequency
            batch_size (`int`):
                Batch size, i.e the number of samples.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            codebook_size (`int`):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`):
                Optional `Bark` speaker prompt.
        Returns: Returns:
            `tuple(torch.FloatTensor)`:
            - **x_semantic_history** (`torch.FloatTensor` -- Processed semantic speaker prompt.
            - **x_coarse_history** (`torch.FloatTensor`) -- Processed coarse speaker prompt.
        """
        if history_prompt is not None:
            x_semantic_history = torch.repeat_interleave(history_prompt['semantic_prompt'][None], batch_size, dim=0)
            x_coarse_history = history_prompt['coarse_prompt'].clone()
            if codebook_size is not None:
                for n in range(1, x_coarse_history.shape[0]):
                    x_coarse_history[n, :] += codebook_size * n
            x_coarse_history = torch.transpose(x_coarse_history, 0, 1).view(-1)
            x_coarse_history = x_coarse_history + semantic_generation_config.semantic_vocab_size
            x_coarse_history = torch.repeat_interleave(x_coarse_history[None], batch_size, dim=0)
            max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
            n_semantic_hist_provided = min([max_semantic_history, x_semantic_history.shape[1] - x_semantic_history.shape[1] % 2, int(np.floor(x_coarse_history.shape[1] / semantic_to_coarse_ratio))])
            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
            x_semantic_history = x_semantic_history[:, -n_semantic_hist_provided:].int()
            x_coarse_history = x_coarse_history[:, -n_coarse_hist_provided:].int()
            x_coarse_history = x_coarse_history[:, :-2]
        else:
            x_semantic_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
            x_coarse_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
        return (x_semantic_history, x_coarse_history)

    def generate(self, semantic_output: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, coarse_generation_config: BarkCoarseGenerationConfig=None, codebook_size: int=1024, history_prompt: Optional[Dict[str, torch.Tensor]]=None, return_output_lengths: Optional[bool]=None, **kwargs) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]:
        """
        Generates coarse acoustics tokens from input text semantic tokens and an additional optional `Bark` speaker
        prompt.

        Args:
            semantic_output (`torch.Tensor` of shape (batch_size, seq_len), *optional*):
                Input text semantic ids, i.e the output of `BarkSemanticModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
            return_output_lengths (`bool`, *optional*):
                Whether or not to return the output lengths. Useful when batching.
        Returns:
            By default:
                torch.LongTensor: Output coarse acoustics tokens.
            If `return_output_lengths=True`:
                `Tuple(torch.Tensor, torch.Tensor): The output coarse acoustics tokens, and the length of each sample
                of the batch.
        """
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        if coarse_generation_config is None:
            raise ValueError('`coarse_generation_config` has to be provided')
        max_coarse_input_length = coarse_generation_config.max_coarse_input_length
        max_coarse_history = coarse_generation_config.max_coarse_history
        sliding_window_len = coarse_generation_config.sliding_window_len
        semantic_output.masked_fill_(semantic_output == semantic_generation_config.semantic_pad_token, coarse_generation_config.coarse_semantic_pad_token)
        semantic_to_coarse_ratio = coarse_generation_config.coarse_rate_hz / semantic_generation_config.semantic_rate_hz * coarse_generation_config.n_coarse_codebooks
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        output_lengths = (semantic_output != coarse_generation_config.coarse_semantic_pad_token).sum(1)
        output_lengths = torch.floor(output_lengths * semantic_to_coarse_ratio / coarse_generation_config.n_coarse_codebooks)
        output_lengths = torch.round(output_lengths * coarse_generation_config.n_coarse_codebooks).int()
        max_generated_len = torch.max(output_lengths).item()
        batch_size = semantic_output.shape[0]
        x_semantic_history, x_coarse = self.preprocess_histories(history_prompt=history_prompt, max_coarse_history=max_coarse_history, semantic_to_coarse_ratio=semantic_to_coarse_ratio, batch_size=batch_size, semantic_generation_config=semantic_generation_config, codebook_size=codebook_size)
        base_semantic_idx = x_semantic_history.shape[1]
        semantic_output = torch.hstack([x_semantic_history, semantic_output])
        n_window_steps = int(np.ceil(max_generated_len / sliding_window_len))
        total_generated_len = 0
        len_coarse_history = x_coarse.shape[1]
        for _ in range(n_window_steps):
            semantic_idx = base_semantic_idx + int(round(total_generated_len / semantic_to_coarse_ratio))
            input_coarse = semantic_output[:, np.max([0, semantic_idx - max_semantic_history]):]
            input_coarse = input_coarse[:, :max_coarse_input_length]
            input_coarse = F.pad(input_coarse, (0, max_coarse_input_length - input_coarse.shape[-1]), 'constant', coarse_generation_config.coarse_semantic_pad_token)
            input_coarse = torch.hstack([input_coarse, torch.tensor([[coarse_generation_config.coarse_infer_token]] * batch_size).to(self.device), x_coarse[:, -max_coarse_history:]])
            alternatingLogitsProcessor = AlternatingCodebooksLogitsProcessor(input_coarse.shape[1], semantic_generation_config.semantic_vocab_size, codebook_size)
            output_coarse = super().generate(input_coarse, logits_processor=[alternatingLogitsProcessor], max_new_tokens=min(sliding_window_len, max_generated_len - total_generated_len), generation_config=coarse_generation_config, **kwargs)
            input_coarse_len = input_coarse.shape[1]
            x_coarse = torch.hstack([x_coarse, output_coarse[:, input_coarse_len:]])
            total_generated_len = x_coarse.shape[1] - len_coarse_history
            del output_coarse
        coarse_output = x_coarse[:, len_coarse_history:]
        if return_output_lengths:
            return (coarse_output, output_lengths)
        return coarse_output