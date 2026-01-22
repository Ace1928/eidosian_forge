import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
def _reorder_stacked(hidden_states, new_order):
    n_docs = hidden_states.shape[0] // new_order.shape[0]
    hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
    hidden_states = hidden_states.index_select(0, new_order)
    result = hidden_states.view(-1, *hidden_states.shape[2:])
    return result