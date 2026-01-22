import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import yaml
from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION
@dataclass
class BaseConfig:
    compute_environment: ComputeEnvironment
    distributed_type: Union[DistributedType, SageMakerDistributedType]
    mixed_precision: str
    use_cpu: bool
    debug: bool

    def to_dict(self):
        result = self.__dict__
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
            if isinstance(value, dict) and (not bool(value)):
                result[key] = None
        result = {k: v for k, v in result.items() if v is not None}
        return result

    @classmethod
    def from_json_file(cls, json_file=None):
        json_file = default_json_config_file if json_file is None else json_file
        with open(json_file, encoding='utf-8') as f:
            config_dict = json.load(f)
        if 'compute_environment' not in config_dict:
            config_dict['compute_environment'] = ComputeEnvironment.LOCAL_MACHINE
        if 'mixed_precision' not in config_dict:
            config_dict['mixed_precision'] = 'fp16' if 'fp16' in config_dict and config_dict['fp16'] else None
        if 'fp16' in config_dict:
            del config_dict['fp16']
        if 'dynamo_backend' in config_dict:
            dynamo_backend = config_dict.pop('dynamo_backend')
            config_dict['dynamo_config'] = {} if dynamo_backend == 'NO' else {'dynamo_backend': dynamo_backend}
        if 'use_cpu' not in config_dict:
            config_dict['use_cpu'] = False
        if 'debug' not in config_dict:
            config_dict['debug'] = False
        extra_keys = sorted(set(config_dict.keys()) - set(cls.__dataclass_fields__.keys()))
        if len(extra_keys) > 0:
            raise ValueError(f'The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `accelerate` version or fix (and potentially remove) these keys from your config file.')
        return cls(**config_dict)

    def to_json_file(self, json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            content = json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'
            f.write(content)

    @classmethod
    def from_yaml_file(cls, yaml_file=None):
        yaml_file = default_yaml_config_file if yaml_file is None else yaml_file
        with open(yaml_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        if 'compute_environment' not in config_dict:
            config_dict['compute_environment'] = ComputeEnvironment.LOCAL_MACHINE
        if 'mixed_precision' not in config_dict:
            config_dict['mixed_precision'] = 'fp16' if 'fp16' in config_dict and config_dict['fp16'] else None
        if isinstance(config_dict['mixed_precision'], bool) and (not config_dict['mixed_precision']):
            config_dict['mixed_precision'] = 'no'
        if 'fp16' in config_dict:
            del config_dict['fp16']
        if 'dynamo_backend' in config_dict:
            dynamo_backend = config_dict.pop('dynamo_backend')
            config_dict['dynamo_config'] = {} if dynamo_backend == 'NO' else {'dynamo_backend': dynamo_backend}
        if 'use_cpu' not in config_dict:
            config_dict['use_cpu'] = False
        if 'debug' not in config_dict:
            config_dict['debug'] = False
        extra_keys = sorted(set(config_dict.keys()) - set(cls.__dataclass_fields__.keys()))
        if len(extra_keys) > 0:
            raise ValueError(f'The config file at {yaml_file} had unknown keys ({extra_keys}), please try upgrading your `accelerate` version or fix (and potentially remove) these keys from your config file.')
        return cls(**config_dict)

    def to_yaml_file(self, yaml_file):
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f)

    def __post_init__(self):
        if isinstance(self.compute_environment, str):
            self.compute_environment = ComputeEnvironment(self.compute_environment)
        if isinstance(self.distributed_type, str):
            if self.compute_environment == ComputeEnvironment.AMAZON_SAGEMAKER:
                self.distributed_type = SageMakerDistributedType(self.distributed_type)
            else:
                self.distributed_type = DistributedType(self.distributed_type)
        if self.dynamo_config is None:
            self.dynamo_config = {}