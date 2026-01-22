from __future__ import print_function
import os
from copy import copy
from enum import Enum
from .. import CatBoostError, CatBoost
from .evaluation_result import EvaluationResults, MetricEvaluationResult
from ._fold_models_handler import FoldModelsHandler
from ._readers import _SimpleStreamingFileReader
from ._splitter import _Splitter
from .execution_case import ExecutionCase
from .factor_utils import LabelMode, FactorUtils
def _calculate_result_metrics(self, cases, metrics, thread_count=-1, evaluation_step=1):
    """
        This method calculate metrics and return them.

        Args:
            :param cases: List of the ExecutionCases you want to evaluate
            :param metrics: List of the metrics to be computed
            :param thread_count: Count of threads to use.
            :param: evaluation_step: Step to evaluate metrics
            :return: instance of EvaluationResult
        """
    cases_set = set(cases)
    if len(cases_set) != len(cases):
        raise CatBoostError('Found duplicate cases in ' + cases)
    current_wd = self.__go_to_working_dir()
    try:
        if self._fold_count <= self._fold_offset:
            error_msg = 'Count of folds(folds_count - offset) need to be at least one: offset {}, folds_count {}.'
            raise AttributeError(error_msg.format(self._fold_offset, self._fold_count))
        handler = FoldModelsHandler(cases=cases, metrics=metrics, eval_step=evaluation_step, thread_count=thread_count, remove_models=self._remove_models)
        reader = _SimpleStreamingFileReader(self._path_to_dataset, sep=self._delimiter, has_header=self._has_header, group_feature_num=self._group_feature_num)
        splitter = _Splitter(reader, self._column_description, seed=self._seed, min_folds_count=self._min_fold_count)
        result = handler.proceed(splitter=splitter, fold_size=self._fold_size, folds_count=self._fold_count, fold_offset=self._fold_offset)
        return self._create_evaluation_results(result)
    finally:
        os.chdir(current_wd)