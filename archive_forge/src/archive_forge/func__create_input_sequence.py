import itertools
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging
def _create_input_sequence(self, text: Union[TextInput], text_pair: Optional[Union[TextInput]]=None, entities: Optional[EntityInput]=None, entities_pair: Optional[EntityInput]=None, entity_spans: Optional[EntitySpanInput]=None, entity_spans_pair: Optional[EntitySpanInput]=None, **kwargs) -> Tuple[list, list, list, list, list, list]:

    def get_input_ids(text):
        tokens = self.tokenize(text, **kwargs)
        return self.convert_tokens_to_ids(tokens)

    def get_input_ids_and_entity_token_spans(text, entity_spans):
        if entity_spans is None:
            return (get_input_ids(text), None)
        cur = 0
        input_ids = []
        entity_token_spans = [None] * len(entity_spans)
        split_char_positions = sorted(frozenset(itertools.chain(*entity_spans)))
        char_pos2token_pos = {}
        for split_char_position in split_char_positions:
            orig_split_char_position = split_char_position
            if split_char_position > 0 and text[split_char_position - 1] == ' ':
                split_char_position -= 1
            if cur != split_char_position:
                input_ids += get_input_ids(text[cur:split_char_position])
                cur = split_char_position
            char_pos2token_pos[orig_split_char_position] = len(input_ids)
        input_ids += get_input_ids(text[cur:])
        entity_token_spans = [(char_pos2token_pos[char_start], char_pos2token_pos[char_end]) for char_start, char_end in entity_spans]
        return (input_ids, entity_token_spans)
    first_ids, second_ids = (None, None)
    first_entity_ids, second_entity_ids = (None, None)
    first_entity_token_spans, second_entity_token_spans = (None, None)
    if self.task is None:
        if entity_spans is None:
            first_ids = get_input_ids(text)
        else:
            self._check_entity_input_format(entities, entity_spans)
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
            if entities is None:
                first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)
            else:
                first_entity_ids = [self.entity_vocab.get(entity, self.entity_unk_token_id) for entity in entities]
        if text_pair is not None:
            if entity_spans_pair is None:
                second_ids = get_input_ids(text_pair)
            else:
                self._check_entity_input_format(entities_pair, entity_spans_pair)
                second_ids, second_entity_token_spans = get_input_ids_and_entity_token_spans(text_pair, entity_spans_pair)
                if entities_pair is None:
                    second_entity_ids = [self.entity_mask_token_id] * len(entity_spans_pair)
                else:
                    second_entity_ids = [self.entity_vocab.get(entity, self.entity_unk_token_id) for entity in entities_pair]
    elif self.task == 'entity_classification':
        if not (isinstance(entity_spans, list) and len(entity_spans) == 1 and isinstance(entity_spans[0], tuple)):
            raise ValueError('Entity spans should be a list containing a single tuple containing the start and end character indices of an entity')
        first_entity_ids = [self.entity_mask_token_id]
        first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
        entity_token_start, entity_token_end = first_entity_token_spans[0]
        first_ids = first_ids[:entity_token_end] + [self.additional_special_tokens_ids[0]] + first_ids[entity_token_end:]
        first_ids = first_ids[:entity_token_start] + [self.additional_special_tokens_ids[0]] + first_ids[entity_token_start:]
        first_entity_token_spans = [(entity_token_start, entity_token_end + 2)]
    elif self.task == 'entity_pair_classification':
        if not (isinstance(entity_spans, list) and len(entity_spans) == 2 and isinstance(entity_spans[0], tuple) and isinstance(entity_spans[1], tuple)):
            raise ValueError('Entity spans should be provided as a list of two tuples, each tuple containing the start and end character indices of an entity')
        head_span, tail_span = entity_spans
        first_entity_ids = [self.entity_mask_token_id, self.entity_mask2_token_id]
        first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
        head_token_span, tail_token_span = first_entity_token_spans
        token_span_with_special_token_ids = [(head_token_span, self.additional_special_tokens_ids[0]), (tail_token_span, self.additional_special_tokens_ids[1])]
        if head_token_span[0] < tail_token_span[0]:
            first_entity_token_spans[0] = (head_token_span[0], head_token_span[1] + 2)
            first_entity_token_spans[1] = (tail_token_span[0] + 2, tail_token_span[1] + 4)
            token_span_with_special_token_ids = reversed(token_span_with_special_token_ids)
        else:
            first_entity_token_spans[0] = (head_token_span[0] + 2, head_token_span[1] + 4)
            first_entity_token_spans[1] = (tail_token_span[0], tail_token_span[1] + 2)
        for (entity_token_start, entity_token_end), special_token_id in token_span_with_special_token_ids:
            first_ids = first_ids[:entity_token_end] + [special_token_id] + first_ids[entity_token_end:]
            first_ids = first_ids[:entity_token_start] + [special_token_id] + first_ids[entity_token_start:]
    elif self.task == 'entity_span_classification':
        if not (isinstance(entity_spans, list) and len(entity_spans) > 0 and isinstance(entity_spans[0], tuple)):
            raise ValueError('Entity spans should be provided as a list of tuples, each tuple containing the start and end character indices of an entity')
        first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
        first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)
    else:
        raise ValueError(f'Task {self.task} not supported')
    return (first_ids, second_ids, first_entity_ids, second_entity_ids, first_entity_token_spans, second_entity_token_spans)