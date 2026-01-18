from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set
def update_flavor_conf_to_persist_pretrained_model(original_flavor_conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the flavor configuration that was saved with save_pretrained=False to the one that
    includes the local path to the model binary file.
    """
    flavor_conf = original_flavor_conf.copy()
    if FlavorKey.MODEL_BINARY in original_flavor_conf:
        raise MlflowException('It appears that the pretrained model weight is already saved to the artifact path.', error_code=ALREADY_EXISTS)
    from mlflow.transformers.model_io import _MODEL_BINARY_FILE_NAME
    flavor_conf[FlavorKey.MODEL_BINARY] = _MODEL_BINARY_FILE_NAME
    flavor_conf.pop(FlavorKey.MODEL_REVISION, None)
    components = original_flavor_conf.get(FlavorKey.COMPONENTS, [])
    if FlavorKey.PROCESSOR_TYPE in original_flavor_conf:
        components.append(FlavorKey.PROCESSOR)
    for component in components:
        flavor_conf.pop(FlavorKey.COMPONENT_NAME.format(component), None)
        flavor_conf.pop(FlavorKey.COMPONENT_REVISION.format(component), None)
    return flavor_conf