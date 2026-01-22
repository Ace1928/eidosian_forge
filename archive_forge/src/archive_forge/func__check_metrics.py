from ...loss import SoftmaxCrossEntropyLoss
from ....metric import Accuracy, EvalMetric, CompositeEvalMetric
def _check_metrics(metrics):
    if isinstance(metrics, CompositeEvalMetric):
        metrics = [m for metric in metrics.metrics for m in _check_metrics(metric)]
    elif isinstance(metrics, EvalMetric):
        metrics = [metrics]
    else:
        metrics = metrics or []
        if not all([isinstance(metric, EvalMetric) for metric in metrics]):
            raise ValueError('metrics must be a Metric or a list of Metric, refer to mxnet.metric.EvalMetric: {}'.format(metrics))
    return metrics