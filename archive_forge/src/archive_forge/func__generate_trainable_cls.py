import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type
from ray import train, tune
from ray._private.dict import flatten_dict
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.constants import MODEL_KEY, TRAIN_DATASET_KEY
from ray.train.trainer import BaseTrainer, GenDataset
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI
def _generate_trainable_cls(self) -> Type['Trainable']:
    trainable_cls = super()._generate_trainable_cls()
    trainer_cls = self.__class__
    scaling_config = self.scaling_config
    ray_params_cls = self._ray_params_cls
    default_ray_params = self._default_ray_params

    class GBDTTrainable(trainable_cls):

        @classmethod
        def default_resource_request(cls, config):
            updated_scaling_config = config.get('scaling_config', scaling_config)
            if isinstance(updated_scaling_config, dict):
                updated_scaling_config = ScalingConfig(**updated_scaling_config)
            validated_scaling_config = trainer_cls._validate_scaling_config(updated_scaling_config)
            return _convert_scaling_config_to_ray_params(validated_scaling_config, ray_params_cls, default_ray_params).get_tune_resources()
    return GBDTTrainable