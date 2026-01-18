import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
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