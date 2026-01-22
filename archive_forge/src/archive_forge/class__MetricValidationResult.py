from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
class _MetricValidationResult:
    """
    Internal class for representing validation result per metric.
    Not user facing, used for organizing metric failures and generating failure message
    more conveniently.

    Args:
        metric_name: String representing the metric name
        candidate_metric_value: value of metric for candidate model
        metric_threshold: :py:class: `MetricThreshold<mlflow.models.validation.MetricThreshold>`
            The MetricThreshold for the metric.
        baseline_metric_value: value of metric for baseline model
    """
    missing_candidate = False
    missing_baseline = False
    threshold_failed = False
    min_absolute_change_failed = False
    min_relative_change_failed = False

    def __init__(self, metric_name, candidate_metric_value, metric_threshold, baseline_metric_value=None):
        self.metric_name = metric_name
        self.candidate_metric_value = candidate_metric_value
        self.baseline_metric_value = baseline_metric_value
        self.metric_threshold = metric_threshold

    def __str__(self):
        """
        Returns a human-readable string representing the validation result for the metric.
        """
        if self.is_success():
            return f'Metric {self.metric_name} passed the validation.'
        if self.missing_candidate:
            return f'Metric validation failed: metric {self.metric_name} was missing from the evaluation result of the candidate model.'
        result_strs = []
        if self.threshold_failed:
            result_strs.append(f'Metric {self.metric_name} value threshold check failed: candidate model {self.metric_name} = {self.candidate_metric_value}, {self.metric_name} threshold = {self.metric_threshold.threshold}.')
        if self.missing_baseline:
            result_strs.append(f'Model comparison failed: metric {self.metric_name} was missing from the evaluation result of the baseline model.')
        else:
            if self.min_absolute_change_failed:
                result_strs.append(f'Metric {self.metric_name} minimum absolute change check failed: candidate model {self.metric_name} = {self.candidate_metric_value}, baseline model {self.metric_name} = {self.baseline_metric_value}, {self.metric_name} minimum absolute change threshold = {self.metric_threshold.min_absolute_change}.')
            if self.min_relative_change_failed:
                result_strs.append(f'Metric {self.metric_name} minimum relative change check failed: candidate model {self.metric_name} = {self.candidate_metric_value}, baseline model {self.metric_name} = {self.baseline_metric_value}, {self.metric_name} minimum relative change threshold = {self.metric_threshold.min_relative_change}.')
        return ' '.join(result_strs)

    def is_success(self):
        return not self.missing_candidate and (not self.missing_baseline) and (not self.threshold_failed) and (not self.min_absolute_change_failed) and (not self.min_relative_change_failed)