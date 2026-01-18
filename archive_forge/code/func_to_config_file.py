from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.text_format import Merge
from ._base import TFSModelConfig, TFSModelVersion, List,  Union, Optional, Any, Dict
from ._base import dataclass, lazyclass
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
@classmethod
def to_config_file(cls, save_path: str, configs: List[Dict[str, Any]]=None, model_configs: Union[ModelServerConfig, List[TFSModelConfig]]=None, as_protobuf: bool=False, **kwargs):
    assert configs or model_configs, 'Configs or ModelConfigs must be provided'
    filepath = get_pathlike(save_path)
    srv_config = model_configs
    if configs:
        srv_config = TFSConfig.from_dict(configs, **kwargs)
    elif not isinstance(model_configs, ModelServerConfig):
        srv_config = TFSConfig.from_model_configs(model_configs, **kwargs)
    filepath.write_bytes(srv_config.SerializeToString()) if as_protobuf else filepath.write_text(str(srv_config))
    return {'config': srv_config, 'file': filepath}