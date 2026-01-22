from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def _checkpoint_artifact(model: Union[CatBoostClassifier, CatBoostRegressor], aliases: List[str]) -> None:
    """Upload model checkpoint as W&B artifact."""
    if wandb.run is None:
        raise wandb.Error('You must call `wandb.init()` before `_checkpoint_artifact()`')
    model_name = f'model_{wandb.run.id}'
    model_path = Path(wandb.run.dir) / 'model'
    model.save_model(model_path)
    model_artifact = wandb.Artifact(name=model_name, type='model')
    model_artifact.add_file(str(model_path))
    wandb.log_artifact(model_artifact, aliases=aliases)