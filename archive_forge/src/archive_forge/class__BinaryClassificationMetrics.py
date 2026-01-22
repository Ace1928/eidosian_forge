import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
class _BinaryClassificationMetrics(object):
    """Private container class for classification metric statistics.

    True/false positive and true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.global_true_positives = 0
        self.global_false_negatives = 0
        self.global_false_positives = 0
        self.global_true_negatives = 0

    def update_binary_stats(self, label, pred):
        """Update various binary classification counts for a single (label, pred) pair.

        Parameters
        ----------
        label : `NDArray`
            The labels of the data.

        pred : `NDArray`
            Predicted values.
        """
        pred = pred.asnumpy()
        label = label.asnumpy().astype('int32')
        pred_label = numpy.argmax(pred, axis=1)
        check_label_shapes(label, pred)
        if len(numpy.unique(label)) > 2:
            raise ValueError('%s currently only supports binary classification.' % self.__class__.__name__)
        pred_true = pred_label == 1
        pred_false = 1 - pred_true
        label_true = label == 1
        label_false = 1 - label_true
        true_pos = (pred_true * label_true).sum()
        false_pos = (pred_true * label_false).sum()
        false_neg = (pred_false * label_true).sum()
        true_neg = (pred_false * label_false).sum()
        self.true_positives += true_pos
        self.global_true_positives += true_pos
        self.false_positives += false_pos
        self.global_false_positives += false_pos
        self.false_negatives += false_neg
        self.global_false_negatives += false_neg
        self.true_negatives += true_neg
        self.global_true_negatives += true_neg

    @property
    def precision(self):
        if self.true_positives + self.false_positives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_positives)
        else:
            return 0.0

    @property
    def global_precision(self):
        if self.global_true_positives + self.global_false_positives > 0:
            return float(self.global_true_positives) / (self.global_true_positives + self.global_false_positives)
        else:
            return 0.0

    @property
    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_negatives)
        else:
            return 0.0

    @property
    def global_recall(self):
        if self.global_true_positives + self.global_false_negatives > 0:
            return float(self.global_true_positives) / (self.global_true_positives + self.global_false_negatives)
        else:
            return 0.0

    @property
    def fscore(self):
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.0

    @property
    def global_fscore(self):
        if self.global_precision + self.global_recall > 0:
            return 2 * self.global_precision * self.global_recall / (self.global_precision + self.global_recall)
        else:
            return 0.0

    def matthewscc(self, use_global=False):
        """Calculate the Matthew's Correlation Coefficent"""
        if use_global:
            if not self.global_total_examples:
                return 0.0
            true_pos = float(self.global_true_positives)
            false_pos = float(self.global_false_positives)
            false_neg = float(self.global_false_negatives)
            true_neg = float(self.global_true_negatives)
        else:
            if not self.total_examples:
                return 0.0
            true_pos = float(self.true_positives)
            false_pos = float(self.false_positives)
            false_neg = float(self.false_negatives)
            true_neg = float(self.true_negatives)
        terms = [true_pos + false_pos, true_pos + false_neg, true_neg + false_pos, true_neg + false_neg]
        denom = 1.0
        for t in filter(lambda t: t != 0.0, terms):
            denom *= t
        return (true_pos * true_neg - false_pos * false_neg) / math.sqrt(denom)

    @property
    def total_examples(self):
        return self.false_negatives + self.false_positives + self.true_negatives + self.true_positives

    @property
    def global_total_examples(self):
        return self.global_false_negatives + self.global_false_positives + self.global_true_negatives + self.global_true_positives

    def local_reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

    def reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.global_false_positives = 0
        self.global_false_negatives = 0
        self.global_true_positives = 0
        self.global_true_negatives = 0