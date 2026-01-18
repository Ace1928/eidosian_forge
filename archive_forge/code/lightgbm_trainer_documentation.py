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
Retrieve the LightGBM model stored in this checkpoint.