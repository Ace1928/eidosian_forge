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
@dataclass
class RayParamsFromScalingConfig(ray_params_cls):
    placement_options: Dict[str, Any] = None

    def get_tune_resources(self) -> PlacementGroupFactory:
        pgf = super().get_tune_resources()
        placement_options = self.placement_options.copy()
        extended_pgf = PlacementGroupFactory(pgf.bundles, **placement_options)
        extended_pgf._head_bundle_is_empty = pgf._head_bundle_is_empty
        return extended_pgf