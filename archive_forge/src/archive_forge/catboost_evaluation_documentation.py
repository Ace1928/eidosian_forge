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
More flexible evaluation of any cases.
            Args:
            baseline_case: Execution case used for baseline
            compare_cases: List of cases to compare
            eval_metrics: Metrics to calculate
            thread_count: thread_count to use.  Will override one in cases
            Returns
            -------
            result : Instance of EvaluationResult class
        