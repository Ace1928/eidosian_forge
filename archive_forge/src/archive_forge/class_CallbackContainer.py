import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
class CallbackContainer:
    """A special internal callback for invoking a list of other callbacks.

    .. versionadded:: 1.3.0

    """

    def __init__(self, callbacks: Sequence[TrainingCallback], metric: Optional[Callable]=None, output_margin: bool=True, is_cv: bool=False) -> None:
        self.callbacks = set(callbacks)
        for cb in callbacks:
            if not isinstance(cb, TrainingCallback):
                raise TypeError('callback must be an instance of `TrainingCallback`.')
        msg = 'metric must be callable object for monitoring.  For builtin metrics, passing them in training parameter invokes monitor automatically.'
        if metric is not None and (not callable(metric)):
            raise TypeError(msg)
        self.metric = metric
        self.history: TrainingCallback.EvalsLog = collections.OrderedDict()
        self._output_margin = output_margin
        self.is_cv = is_cv
        if self.is_cv:
            self.aggregated_cv = None

    def before_training(self, model: _Model) -> _Model:
        """Function called before training."""
        for c in self.callbacks:
            model = c.before_training(model=model)
            msg = 'before_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def after_training(self, model: _Model) -> _Model:
        """Function called after training."""
        for c in self.callbacks:
            model = c.after_training(model=model)
            msg = 'after_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def before_iteration(self, model: _Model, epoch: int, dtrain: DMatrix, evals: Optional[List[Tuple[DMatrix, str]]]) -> bool:
        """Function called before training iteration."""
        return any((c.before_iteration(model, epoch, self.history) for c in self.callbacks))

    def _update_history(self, score: Union[List[Tuple[str, float]], List[Tuple[str, float, float]]], epoch: int) -> None:
        for d in score:
            name: str = d[0]
            s: float = d[1]
            if self.is_cv:
                std = float(cast(Tuple[str, float, float], d)[2])
                x: _Score = (s, std)
            else:
                x = s
            splited_names = name.split('-')
            data_name = splited_names[0]
            metric_name = '-'.join(splited_names[1:])
            x = _allreduce_metric(x)
            if data_name not in self.history:
                self.history[data_name] = collections.OrderedDict()
            data_history = self.history[data_name]
            if metric_name not in data_history:
                data_history[metric_name] = cast(_ScoreList, [])
            metric_history = data_history[metric_name]
            if self.is_cv:
                cast(List[Tuple[float, float]], metric_history).append(cast(Tuple[float, float], x))
            else:
                cast(List[float], metric_history).append(cast(float, x))

    def after_iteration(self, model: _Model, epoch: int, dtrain: DMatrix, evals: Optional[List[Tuple[DMatrix, str]]]) -> bool:
        """Function called after training iteration."""
        if self.is_cv:
            scores = model.eval(epoch, self.metric, self._output_margin)
            scores = _aggcv(scores)
            self.aggregated_cv = scores
            self._update_history(scores, epoch)
        else:
            evals = [] if evals is None else evals
            for _, name in evals:
                assert name.find('-') == -1, 'Dataset name should not contain `-`'
            score: str = model.eval_set(evals, epoch, self.metric, self._output_margin)
            metric_score = _parse_eval_str(score)
            self._update_history(metric_score, epoch)
        ret = any((c.after_iteration(model, epoch, self.history) for c in self.callbacks))
        return ret