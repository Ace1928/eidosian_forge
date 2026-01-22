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
@property
def _ray_params(self) -> 'xgboost_ray.RayParams':
    scaling_config_dataclass = self._validate_scaling_config(self.scaling_config)
    return _convert_scaling_config_to_ray_params(scaling_config_dataclass, self._ray_params_cls, self._default_ray_params)