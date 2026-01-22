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
def _check_entity_input_format(self, entities: Optional[EntityInput], entity_spans: Optional[EntitySpanInput]):
    if not isinstance(entity_spans, list):
        raise ValueError('entity_spans should be given as a list')
    elif len(entity_spans) > 0 and (not isinstance(entity_spans[0], tuple)):
        raise ValueError('entity_spans should be given as a list of tuples containing the start and end character indices')
    if entities is not None:
        if not isinstance(entities, list):
            raise ValueError('If you specify entities, they should be given as a list')
        if len(entities) > 0 and (not isinstance(entities[0], str)):
            raise ValueError('If you specify entities, they should be given as a list of entity names')
        if len(entities) != len(entity_spans):
            raise ValueError('If you specify entities, entities and entity_spans must be the same length')