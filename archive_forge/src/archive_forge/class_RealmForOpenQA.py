import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
@add_start_docstrings('`RealmForOpenQA` for end-to-end open domain question answering.', REALM_START_DOCSTRING)
class RealmForOpenQA(RealmPreTrainedModel):

    def __init__(self, config, retriever=None):
        super().__init__(config)
        self.embedder = RealmEmbedder(config)
        self.reader = RealmReader(config)
        self.register_buffer('block_emb', torch.zeros(()).new_empty(size=(config.num_block_records, config.retriever_proj_size), dtype=torch.float32, device=torch.device('cpu')))
        self.retriever = retriever
        self.post_init()

    @property
    def searcher_beam_size(self):
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    def block_embedding_to(self, device):
        """Send `self.block_emb` to a specific device.

        Args:
            device (`str` or `torch.device`):
                The device to which `self.block_emb` will be sent.
        """
        self.block_emb = self.block_emb.to(device)

    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format('1, sequence_length'))
    @replace_return_docstrings(output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor], attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, answer_ids: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmForOpenQAOutput]:
        """
        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import RealmForOpenQA, RealmRetriever, AutoTokenizer

        >>> retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
        >>> model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever)

        >>> question = "Who is the pioneer in modern computer science?"
        >>> question_ids = tokenizer([question], return_tensors="pt")
        >>> answer_ids = tokenizer(
        ...     ["alan mathison turing"],
        ...     add_special_tokens=False,
        ...     return_token_type_ids=False,
        ...     return_attention_mask=False,
        ... ).input_ids

        >>> reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)
        >>> predicted_answer = tokenizer.decode(predicted_answer_ids)
        >>> loss = reader_output.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError('The batch_size of the inputs must be 1.')
        question_outputs = self.embedder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True)
        question_projection = question_outputs[0]
        batch_scores = torch.einsum('BD,QD->QB', self.block_emb, question_projection.to(self.block_emb.device))
        _, retrieved_block_ids = torch.topk(batch_scores, k=self.searcher_beam_size, dim=-1)
        retrieved_block_ids = retrieved_block_ids.squeeze()
        retrieved_block_emb = torch.index_select(self.block_emb, dim=0, index=retrieved_block_ids)
        has_answers, start_pos, end_pos, concat_inputs = self.retriever(retrieved_block_ids.cpu(), input_ids, answer_ids, max_length=self.config.reader_seq_len)
        concat_inputs = concat_inputs.to(self.reader.device)
        block_mask = concat_inputs.special_tokens_mask.type(torch.bool).to(device=self.reader.device)
        block_mask.logical_not_().logical_and_(concat_inputs.token_type_ids.type(torch.bool))
        if has_answers is not None:
            has_answers = torch.tensor(has_answers, dtype=torch.bool, device=self.reader.device)
            start_pos = torch.tensor(start_pos, dtype=torch.long, device=self.reader.device)
            end_pos = torch.tensor(end_pos, dtype=torch.long, device=self.reader.device)
        retrieved_logits = torch.einsum('D,BD->B', question_projection.squeeze(), retrieved_block_emb.to(self.reader.device))
        reader_output = self.reader(input_ids=concat_inputs.input_ids[0:self.config.reader_beam_size], attention_mask=concat_inputs.attention_mask[0:self.config.reader_beam_size], token_type_ids=concat_inputs.token_type_ids[0:self.config.reader_beam_size], relevance_score=retrieved_logits, block_mask=block_mask, has_answers=has_answers, start_positions=start_pos, end_positions=end_pos, return_dict=True)
        predicted_block = concat_inputs.input_ids[reader_output.block_idx]
        predicted_answer_ids = predicted_block[reader_output.start_pos:reader_output.end_pos + 1]
        if not return_dict:
            return (reader_output, predicted_answer_ids)
        return RealmForOpenQAOutput(reader_output=reader_output, predicted_answer_ids=predicted_answer_ids)