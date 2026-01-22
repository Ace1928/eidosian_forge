from parlai.core.opt import Opt
from parlai.utils.torch import PipelineHelper
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.metrics import Metric, AverageMetric
from typing import List, Optional, Tuple, Dict
from parlai.utils.typing import TScalar
import parlai.utils.logging as logging
import torch
import torch.nn.functional as F
class WeightedF1Metric(Metric):
    """
    Class that represents the weighted f1 from ClassificationF1Metric.
    """
    __slots__ = '_values'

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, metrics: Dict[str, ClassificationF1Metric]) -> None:
        self._values: Dict[str, ClassificationF1Metric] = metrics

    def __add__(self, other: Optional['WeightedF1Metric']) -> 'WeightedF1Metric':
        if other is None:
            return self
        assert isinstance(other, WeightedF1Metric)
        output: Dict[str, ClassificationF1Metric] = dict(**self._values)
        for k, v in other._values.items():
            output[k] = output.get(k, None) + v
        return WeightedF1Metric(output)

    def value(self) -> float:
        weighted_f1 = 0.0
        values = list(self._values.values())
        if len(values) == 0:
            return weighted_f1
        total_examples = values[0]._true_positives + values[0]._true_negatives + values[0]._false_positives + values[0]._false_negatives
        for each in values:
            actual_positive = each._true_positives + each._false_negatives
            weighted_f1 += each.value() * (actual_positive / total_examples)
        return weighted_f1

    @staticmethod
    def compute_many(metrics: Dict[str, List[ClassificationF1Metric]]) -> List['WeightedF1Metric']:
        weighted_f1s = [dict(zip(metrics, t)) for t in zip(*metrics.values())]
        return [WeightedF1Metric(metrics) for metrics in weighted_f1s]