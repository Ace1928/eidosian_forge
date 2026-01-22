import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast
import xgboost as xgb  # type: ignore
from xgboost import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def after_training(self, model: Booster) -> Booster:
    """Run after training is finished."""
    if self.log_model:
        self._log_model_as_artifact(model)
    if self.log_feature_importance:
        self._log_feature_importance(model)
    if model.attr('best_score') is not None:
        wandb.log({'best_score': float(cast(str, model.attr('best_score'))), 'best_iteration': int(cast(str, model.attr('best_iteration')))})
    return model