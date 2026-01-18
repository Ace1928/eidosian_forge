import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def tf_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import tensorflow as tf
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    if 'label' in first and first['label'] is not None:
        label_col_name = 'label'
    elif 'label_ids' in first and first['label_ids'] is not None:
        label_col_name = 'label_ids'
    elif 'labels' in first and first['labels'] is not None:
        label_col_name = 'labels'
    else:
        label_col_name = None
    if label_col_name is not None:
        if isinstance(first[label_col_name], tf.Tensor):
            dtype = tf.int64 if first[label_col_name].dtype.is_integer else tf.float32
        elif isinstance(first[label_col_name], np.ndarray) or isinstance(first[label_col_name], np.generic):
            dtype = tf.int64 if np.issubdtype(first[label_col_name].dtype, np.integer) else tf.float32
        elif isinstance(first[label_col_name], (tuple, list)):
            dtype = tf.int64 if isinstance(first[label_col_name][0], int) else tf.float32
        else:
            dtype = tf.int64 if isinstance(first[label_col_name], int) else tf.float32
        batch['labels'] = tf.convert_to_tensor([f[label_col_name] for f in features], dtype=dtype)
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'labels') and v is not None and (not isinstance(v, str)):
            if isinstance(v, (tf.Tensor, np.ndarray)):
                batch[k] = tf.stack([f[k] for f in features])
            else:
                batch[k] = tf.convert_to_tensor([f[k] for f in features])
    return batch