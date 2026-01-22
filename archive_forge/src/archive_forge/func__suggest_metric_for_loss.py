from ...loss import SoftmaxCrossEntropyLoss
from ....metric import Accuracy, EvalMetric, CompositeEvalMetric
def _suggest_metric_for_loss(loss):
    if isinstance(loss, SoftmaxCrossEntropyLoss):
        return Accuracy()
    return None