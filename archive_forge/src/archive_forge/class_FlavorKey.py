from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set
class FlavorKey:
    TASK = 'task'
    INSTANCE_TYPE = 'instance_type'
    TORCH_DTYPE = 'torch_dtype'
    FRAMEWORK = 'framework'
    MODEL = 'model'
    MODEL_TYPE = 'pipeline_model_type'
    MODEL_BINARY = 'model_binary'
    MODEL_NAME = 'source_model_name'
    MODEL_REVISION = 'source_model_revision'
    PEFT = 'peft_adaptor'
    COMPONENTS = 'components'
    COMPONENT_NAME = '{}_name'
    COMPONENT_REVISION = '{}_revision'
    COMPONENT_TYPE = '{}_type'
    TOKENIZER = 'tokenizer'
    FEATURE_EXTRACTOR = 'feature_extractor'
    IMAGE_PROCESSOR = 'image_processor'
    PROCESSOR = 'processor'
    PROCESSOR_TYPE = 'processor_type'
    PROMPT_TEMPLATE = 'prompt_template'