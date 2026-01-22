import operator
from typing import List
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from ..fuser_method_mappings import (
from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
from .qnnpack import (
def _get_embedding_op_configs() -> List[BackendPatternConfig]:
    dtype_configs = [executorch_weight_only_quint8_dtype_config]
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [(nn.Embedding, nnqat.Embedding, nnqr.Embedding), (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag)]:
        embedding_op_configs.append(BackendPatternConfig(embedding_op).set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT).set_dtype_configs(dtype_configs).set_qat_module(qat_embedding_op).set_root_module(embedding_op).set_reference_quantized_module(ref_embedding_op))
        embedding_op_configs.append(BackendPatternConfig(qat_embedding_op).set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT).set_dtype_configs(dtype_configs).set_root_module(embedding_op).set_reference_quantized_module(ref_embedding_op))
        embedding_op_configs.append(BackendPatternConfig(torch.nn.functional.embedding).set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1}))
    return embedding_op_configs