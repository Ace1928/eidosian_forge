from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.text_format import Merge
from ._base import TFSModelConfig, TFSModelVersion, List,  Union, Optional, Any, Dict
from ._base import dataclass, lazyclass
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
class TFSConfig:

    @classmethod
    def from_model_configs(cls, model_configs: List[TFSModelConfig]) -> ModelServerConfig:
        serv_config = ModelServerConfig()
        model_config_list = serv_config.model_config_list
        for config in model_configs:
            model = model_config_list.config.add()
            model.name = config.name
            model.base_path = config.base_path
            model.model_platform = config.model_platform
            if config.model_versions:
                labels = model.version_labels
                policy = model.model_version_policy
                versions = policy.specific.versions
                for model_version in config.model_versions:
                    versions.append(model_version.step)
                    labels[model_version.label] = model_version.step
        return serv_config

    @classmethod
    def from_config_file(cls, filepath: str, as_obj: bool=True, **kwargs) -> Union[ModelServerConfig, List[TFSModelConfig]]:
        filepath = get_pathlike(filepath)
        assert filepath.exists()
        serv_config = ModelServerConfig()
        serv_config = Merge(filepath.read_bytes(), serv_config)
        if not as_obj:
            return serv_config
        model_configs = []
        for model in serv_config.model_config_list.config:
            versions = [TFSModelVersion(step=v, label=k) for k, v in model.version_labels.items()]
            model_config = TFSModelConfig(name=model.name, base_path=model.base_path, model_platform=model.model_platform, model_versions=versions)
            model_configs.append(model_config)
        return model_configs

    @classmethod
    def from_protobuf(cls, filepath: str, as_obj: bool=True, **kwargs) -> Union[ModelServerConfig, List[TFSModelConfig]]:
        filepath = get_pathlike(filepath)
        assert filepath.exists()
        serv_config = ModelServerConfig()
        serv_config.FromString(filepath.read_bytes())
        if not as_obj:
            return serv_config
        model_configs = []
        for model in serv_config.model_config_list.config:
            versions = [TFSModelVersion(step=v, label=k) for k, v in model.version_labels.items()]
            model_config = TFSModelConfig(name=model.name, base_path=model.base_path, model_platform=model.model_platform, model_versions=versions)
            model_configs.append(model_config)
        return model_configs

    @classmethod
    def from_dict(cls, configs: List[Dict[str, Any]], exclude=[], include=[], only_latest=False, return_tfconfig=False) -> Union[ModelServerConfig, List[TFSModelConfig]]:
        if exclude:
            configs = [c for c in configs if c.get('name', '') not in exclude]
        if include:
            configs = [c for c in configs if c.get('name', '') in include]
        model_configs = [ModelConfig.from_dict(config) for config in configs]
        if only_latest:
            for config in model_configs:
                config.set_only_latest()
        if not return_tfconfig:
            return model_configs
        return TFSConfig.from_model_configs(model_configs)

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