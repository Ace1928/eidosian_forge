from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
Unflattens the first, flat batch*beam dimension of a non-scalar array.