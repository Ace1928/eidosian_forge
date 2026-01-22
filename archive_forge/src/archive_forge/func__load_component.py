import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def _load_component(flavor_conf, name, local_path=None):
    import transformers
    _COMPONENT_TO_AUTOCLASS_MAP = {FlavorKey.TOKENIZER: transformers.AutoTokenizer, FlavorKey.FEATURE_EXTRACTOR: transformers.AutoFeatureExtractor, FlavorKey.PROCESSOR: transformers.AutoProcessor, FlavorKey.IMAGE_PROCESSOR: transformers.AutoImageProcessor}
    component_name = flavor_conf[FlavorKey.COMPONENT_TYPE.format(name)]
    if hasattr(transformers, component_name):
        cls = getattr(transformers, component_name)
        trust_remote = False
    else:
        if local_path is None:
            raise MlflowException(f'A custom component `{component_name}` was specified, but no local config file was found to retrieve the definition. Make sure your model was saved with save_pretrained=True.')
        cls = _COMPONENT_TO_AUTOCLASS_MAP[name]
        trust_remote = True
    if local_path is not None:
        path = local_path.joinpath(_COMPONENTS_BINARY_DIR_NAME, name)
        return cls.from_pretrained(str(path), trust_remote_code=trust_remote)
    else:
        repo = flavor_conf[FlavorKey.COMPONENT_NAME.format(name)]
        revision = flavor_conf.get(FlavorKey.COMPONENT_REVISION.format(name))
        return cls.from_pretrained(repo, revision=revision, trust_remote_code=trust_remote)