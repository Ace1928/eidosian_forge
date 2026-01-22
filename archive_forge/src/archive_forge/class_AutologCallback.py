import logging
import xgboost
from packaging.version import Version
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass
class AutologCallback(xgboost.callback.TrainingCallback, metaclass=ExceptionSafeAbstractClass):

    def __init__(self, metrics_logger, eval_results):
        self.metrics_logger = metrics_logger
        self.eval_results = eval_results

    def after_iteration(self, model, epoch, evals_log):
        """
            Run after each iteration. Return True when training should stop.
            """
        evaluation_result_dict = {}
        for data_name, metric_dict in evals_log.items():
            metric_dict = _patch_metric_names(metric_dict)
            for metric_name, metric_values_on_each_iter in metric_dict.items():
                key = f'{data_name}-{metric_name}'
                evaluation_result_dict[key] = metric_values_on_each_iter[-1]
        self.metrics_logger.record_metrics(evaluation_result_dict, epoch)
        self.eval_results.append(evaluation_result_dict)
        return False