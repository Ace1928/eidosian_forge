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
class ConfusionMatrixMetric(Metric):
    """
    Class that keeps count of the confusion matrix for classification.

    Also provides helper methods computes precision, recall, f1, weighted_f1 for
    classification.
    """
    __slots__ = ('_true_positives', '_true_negatives', '_false_positives', '_false_negatives')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, true_positives: TScalar=0, true_negatives: TScalar=0, false_positives: TScalar=0, false_negatives: TScalar=0) -> None:
        self._true_positives = self.as_number(true_positives)
        self._true_negatives = self.as_number(true_negatives)
        self._false_positives = self.as_number(false_positives)
        self._false_negatives = self.as_number(false_negatives)

    def __add__(self, other: Optional['ConfusionMatrixMetric']) -> 'ConfusionMatrixMetric':
        if other is None:
            return self
        assert isinstance(other, ConfusionMatrixMetric)
        full_true_positives: TScalar = self._true_positives + other._true_positives
        full_true_negatives: TScalar = self._true_negatives + other._true_negatives
        full_false_positives: TScalar = self._false_positives + other._false_positives
        full_false_negatives: TScalar = self._false_negatives + other._false_negatives
        return type(self)(true_positives=full_true_positives, true_negatives=full_true_negatives, false_positives=full_false_positives, false_negatives=full_false_negatives)

    @staticmethod
    def compute_many(true_positives: TScalar=0, true_negatives: TScalar=0, false_positives: TScalar=0, false_negatives: TScalar=0) -> Tuple['PrecisionMetric', 'RecallMetric', 'ClassificationF1Metric']:
        return (PrecisionMetric(true_positives, true_negatives, false_positives, false_negatives), RecallMetric(true_positives, true_negatives, false_positives, false_negatives), ClassificationF1Metric(true_positives, true_negatives, false_positives, false_negatives))

    @staticmethod
    def compute_metrics(predictions: List[str], gold_labels: List[str], positive_class: str) -> Tuple[List['PrecisionMetric'], List['RecallMetric'], List['ClassificationF1Metric']]:
        precisions = []
        recalls = []
        f1s = []
        for predicted, gold_label in zip(predictions, gold_labels):
            true_positives = int(predicted == positive_class and gold_label == positive_class)
            true_negatives = int(predicted != positive_class and gold_label != positive_class)
            false_positives = int(predicted == positive_class and gold_label != positive_class)
            false_negatives = int(predicted != positive_class and gold_label == positive_class)
            precision, recall, f1 = ConfusionMatrixMetric.compute_many(true_positives, true_negatives, false_positives, false_negatives)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return (precisions, recalls, f1s)