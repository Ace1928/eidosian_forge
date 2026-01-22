import os
from typing import Any, Dict, Union
import lightgbm
import lightgbm_ray
import xgboost_ray
from lightgbm_ray.tune import TuneReportCheckpointCallback
from ray.train import Checkpoint
from ray.train.gbdt_trainer import GBDTTrainer
from ray.train.lightgbm import LightGBMCheckpoint
from ray.util.annotations import PublicAPI
def _load_checkpoint(self, checkpoint: Checkpoint) -> lightgbm.Booster:
    return self.__class__.get_model(checkpoint)