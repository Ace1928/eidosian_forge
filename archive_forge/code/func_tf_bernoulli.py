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
@staticmethod
def tf_bernoulli(shape, probability):
    import tensorflow as tf
    prob_matrix = tf.fill(shape, probability)
    return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)