import os
from typing import Any, Dict
import xgboost
import xgboost_ray
from xgboost_ray.tune import TuneReportCheckpointCallback
from ray.train import Checkpoint
from ray.train.gbdt_trainer import GBDTTrainer
from ray.train.xgboost import XGBoostCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the XGBoost model stored in this checkpoint.