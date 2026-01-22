import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class Metric(object):
    """The base class of metric."""
    __slots__ = ['_metric', '_metric_name', '_metric_methods', '_label_length']

    def __init__(self, metric_name, metric_methods, label_length, *args):
        """Creates a new metric.

    Args:
      metric_name: name of the metric class.
      metric_methods: list of swig metric methods.
      label_length: length of label args.
      *args: the arguments to call create method.
    """
        self._metric_name = metric_name
        self._metric_methods = metric_methods
        self._label_length = label_length
        if label_length >= len(self._metric_methods):
            raise ValueError('Cannot create {} metric with label >= {}'.format(self._metric_name, len(self._metric_methods)))
        self._metric = self._metric_methods[self._label_length].create(*args)

    def __del__(self):
        try:
            deleter = self._metric_methods[self._label_length].delete
            metric = self._metric
        except AttributeError:
            return
        if deleter is not None:
            deleter(metric)

    def get_cell(self, *labels):
        """Retrieves the cell."""
        if len(labels) != self._label_length:
            raise ValueError('The {} expects taking {} labels'.format(self._metric_name, self._label_length))
        return self._metric_methods[self._label_length].get_cell(self._metric, *labels)