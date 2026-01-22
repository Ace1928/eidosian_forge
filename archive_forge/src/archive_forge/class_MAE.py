import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class MAE(EvalMetric):
    """Computes Mean Absolute Error (MAE) loss.

    The mean absolute error is given by

    .. math::
        \\frac{\\sum_i^n |y_i - \\hat{y}_i|}{n}

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> mean_absolute_error = mx.metric.MAE()
    >>> mean_absolute_error.update(labels = labels, preds = predicts)
    >>> print mean_absolute_error.get()
    ('mae', 0.5)
    """

    def __init__(self, name='mae', output_names=None, label_names=None):
        super(MAE, self).__init__(name, output_names=output_names, label_names=label_names, has_global_stats=True)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)
            mae = numpy.abs(label - pred).mean()
            self.sum_metric += mae
            self.global_sum_metric += mae
            self.num_inst += 1
            self.global_num_inst += 1