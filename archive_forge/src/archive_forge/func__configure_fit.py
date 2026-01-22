import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def _configure_fit(self, booster: Optional[Union[Booster, 'XGBModel', str]], eval_metric: Optional[Union[Callable, str, Sequence[str]]], params: Dict[str, Any], early_stopping_rounds: Optional[int], callbacks: Optional[Sequence[TrainingCallback]]) -> Tuple[Optional[Union[Booster, str, 'XGBModel']], Optional[Metric], Dict[str, Any], Optional[int], Optional[Sequence[TrainingCallback]]]:
    """Configure parameters for :py:meth:`fit`."""
    if isinstance(booster, XGBModel):
        model: Optional[Union[Booster, str]] = booster.get_booster()
    else:
        model = booster

    def _deprecated(parameter: str) -> None:
        warnings.warn(f'`{parameter}` in `fit` method is deprecated for better compatibility with scikit-learn, use `{parameter}` in constructor or`set_params` instead.', UserWarning)

    def _duplicated(parameter: str) -> None:
        raise ValueError(f'2 different `{parameter}` are provided.  Use the one in constructor or `set_params` instead.')
    if eval_metric is not None:
        _deprecated('eval_metric')
    if self.eval_metric is not None and eval_metric is not None:
        _duplicated('eval_metric')
    if self.eval_metric is not None:
        from_fit = False
        eval_metric = self.eval_metric
    else:
        from_fit = True
    metric: Optional[Metric] = None
    if eval_metric is not None:
        if callable(eval_metric) and from_fit:
            metric = eval_metric
        elif callable(eval_metric):
            if self._get_type() == 'ranker':
                metric = ltr_metric_decorator(eval_metric, self.n_jobs)
            else:
                metric = _metric_decorator(eval_metric)
        else:
            params.update({'eval_metric': eval_metric})
    if early_stopping_rounds is not None:
        _deprecated('early_stopping_rounds')
    if early_stopping_rounds is not None and self.early_stopping_rounds is not None:
        _duplicated('early_stopping_rounds')
    early_stopping_rounds = self.early_stopping_rounds if self.early_stopping_rounds is not None else early_stopping_rounds
    if callbacks is not None:
        _deprecated('callbacks')
    if callbacks is not None and self.callbacks is not None:
        _duplicated('callbacks')
    callbacks = self.callbacks if self.callbacks is not None else callbacks
    tree_method = params.get('tree_method', None)
    if self.enable_categorical and tree_method == 'exact':
        raise ValueError('Experimental support for categorical data is not implemented for current tree method yet.')
    return (model, metric, params, early_stopping_rounds, callbacks)