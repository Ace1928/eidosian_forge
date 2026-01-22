from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
class _RecordEvaluationCallback:
    """Internal record evaluation callable class."""

    def __init__(self, eval_result: _EvalResultDict) -> None:
        self.order = 20
        self.before_iteration = False
        if not isinstance(eval_result, dict):
            raise TypeError('eval_result should be a dictionary')
        self.eval_result = eval_result

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None:
            raise RuntimeError('record_evaluation() callback enabled but no evaluation results found. This is a probably bug in LightGBM. Please report it at https://github.com/microsoft/LightGBM/issues')
        self.eval_result.clear()
        for item in env.evaluation_result_list:
            if len(item) == 4:
                data_name, eval_name = item[:2]
            else:
                data_name, eval_name = item[1].split()
            self.eval_result.setdefault(data_name, OrderedDict())
            if len(item) == 4:
                self.eval_result[data_name].setdefault(eval_name, [])
            else:
                self.eval_result[data_name].setdefault(f'{eval_name}-mean', [])
                self.eval_result[data_name].setdefault(f'{eval_name}-stdv', [])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if env.evaluation_result_list is None:
            raise RuntimeError('record_evaluation() callback enabled but no evaluation results found. This is a probably bug in LightGBM. Please report it at https://github.com/microsoft/LightGBM/issues')
        for item in env.evaluation_result_list:
            if len(item) == 4:
                data_name, eval_name, result = item[:3]
                self.eval_result[data_name][eval_name].append(result)
            else:
                data_name, eval_name = item[1].split()
                res_mean = item[2]
                res_stdv = item[4]
                self.eval_result[data_name][f'{eval_name}-mean'].append(res_mean)
                self.eval_result[data_name][f'{eval_name}-stdv'].append(res_stdv)