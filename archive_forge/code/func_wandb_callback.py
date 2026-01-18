from pathlib import Path
from typing import TYPE_CHECKING, Callable
import lightgbm  # type: ignore
from lightgbm import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def wandb_callback(log_params: bool=True, define_metric: bool=True) -> Callable:
    """Automatically integrates LightGBM with wandb.

    Arguments:
        log_params: (boolean) if True (default) logs params passed to lightgbm.train as W&B config
        define_metric: (boolean) if True (default) capture model performance at the best step, instead of the last step, of training in your `wandb.summary`

    Passing `wandb_callback` to LightGBM will:
      - log params passed to lightgbm.train as W&B config (default).
      - log evaluation metrics collected by LightGBM, such as rmse, accuracy etc to Weights & Biases
      - Capture the best metric in `wandb.summary` when `define_metric=True` (default).

    Use `log_summary` as an extension of this callback.

    Example:
        ```python
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
        }
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=10,
            valid_sets=lgb_eval,
            valid_names=("validation"),
            callbacks=[wandb_callback()],
        )
        ```
    """
    return _WandbCallback(define_metric)