import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _get_row_updated_score(row_inputs: Tuple[tf.Tensor]) -> tf.Tensor:
    row_input_ids, row_score = row_inputs
    banned_tokens = self._calc_row_banned_bad_tokens(row_input_ids[:cur_len])
    banned_tokens_mask = tf.scatter_nd(indices=tf.expand_dims(banned_tokens, axis=-1), updates=tf.ones_like(banned_tokens, dtype=tf.bool), shape=row_score.shape)
    row_score = tf.where(banned_tokens_mask, -float('inf'), row_score)
    return row_score