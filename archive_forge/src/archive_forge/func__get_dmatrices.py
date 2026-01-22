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
def _get_dmatrices(self, dmatrix_params: Dict[str, Any]) -> Dict[str, 'xgboost_ray.RayDMatrix']:
    return {k: self._dmatrix_cls(v, label=self.label_column, **dmatrix_params.get(k, {})) for k, v in self.datasets.items()}