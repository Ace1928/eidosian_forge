import copy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
class MetricsContainer(Container):
    """A container class for metrics passed to `Model.compile`."""

    def __init__(self, metrics=None, weighted_metrics=None, output_names=None, from_serialized=False):
        """Initializes a container for metrics.

    Arguments:
      metrics: see the `metrics` argument from `tf.keras.Model.compile`.
      weighted_metrics: see the `weighted_metrics` argument from
        `tf.keras.Model.compile`.
      output_names: A list of strings of names of outputs for the model.
      from_serialized: Whether the model being compiled is from a serialized
        model.  Used to avoid redundantly applying pre-processing renaming
        steps.
    """
        super(MetricsContainer, self).__init__(output_names=output_names)
        self._user_metrics = metrics
        self._user_weighted_metrics = weighted_metrics
        self._metrics = metrics
        self._weighted_metrics = weighted_metrics
        self._built = False
        self._from_serialized = from_serialized

    @property
    def metrics(self):
        """All metrics in this container."""
        if not self._built:
            return []
        return self._metrics_in_order

    @property
    def unweighted_metrics(self):
        """Metrics in this container that should not be passed `sample_weight`."""
        if not self._built:
            return None
        return nest.flatten(self._metrics)

    @property
    def weighted_metrics(self):
        """Metrics in this container that should be passed `sample_weight`."""
        if not self._built:
            return None
        return nest.flatten(self._weighted_metrics)

    def build(self, y_pred, y_true):
        """One-time setup of metric objects."""
        super(MetricsContainer, self).build(y_pred)
        self._metrics = self._maybe_broadcast_to_outputs(y_pred, self._metrics)
        self._metrics = self._conform_to_outputs(y_pred, self._metrics)
        self._weighted_metrics = self._maybe_broadcast_to_outputs(y_pred, self._weighted_metrics)
        self._weighted_metrics = self._conform_to_outputs(y_pred, self._weighted_metrics)
        y_pred = nest.list_to_tuple(y_pred)
        y_true = nest.list_to_tuple(y_true)
        self._metrics = nest.list_to_tuple(self._metrics)
        self._weighted_metrics = nest.list_to_tuple(self._weighted_metrics)
        self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects, self._metrics, y_true, y_pred)
        self._weighted_metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects, self._weighted_metrics, y_true, y_pred)
        self._metrics = nest.flatten_up_to(y_pred, self._metrics, check_types=False)
        self._weighted_metrics = nest.flatten_up_to(y_pred, self._weighted_metrics, check_types=False)
        if not self._from_serialized:
            self._set_metric_names()
        self._create_ordered_metrics()
        self._built = True

    @property
    def built(self):
        return self._built

    def _set_metric_names(self):
        """Sets unique metric names."""
        metric_names = set()
        is_multi_output = len(self._output_names) > 1
        zip_args = (self._output_names, self._metrics, self._weighted_metrics)
        for output_name, output_metrics, weighted_output_metrics in zip(*zip_args):
            for m in output_metrics:
                if m is None:
                    continue
                if is_multi_output:
                    m._name = output_name + '_' + m._name
                if m._name in metric_names:
                    raise ValueError('Found two metrics with the same name: {}'.format(m._name))
                metric_names.add(m._name)
            for wm in weighted_output_metrics:
                if wm is None:
                    continue
                if is_multi_output:
                    if output_name + '_' + wm._name in metric_names:
                        wm._name = output_name + '_weighted_' + wm._name
                    else:
                        wm._name = output_name + '_' + wm._name
                elif wm._name in metric_names:
                    wm._name = 'weighted_' + wm._name
                if wm._name in metric_names:
                    raise ValueError('Found two metrics with the same name: {}'.format(wm._name))
                metric_names.add(wm._name)

    def _create_ordered_metrics(self):
        """Cache the flat order needed when returning metrics, for backwards compat."""
        self._metrics_in_order = []
        for output_metrics, output_weighted_metrics in zip(self._metrics, self._weighted_metrics):
            for m in nest.flatten(output_metrics):
                if m is not None:
                    self._metrics_in_order.append(m)
            for wm in nest.flatten(output_weighted_metrics):
                if wm is not None:
                    self._metrics_in_order.append(wm)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state of per-output metrics."""
        y_true = self._conform_to_outputs(y_pred, y_true)
        sample_weight = self._conform_to_outputs(y_pred, sample_weight)
        if not self._built:
            self.build(y_pred, y_true)
        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true) if y_true is not None else []
        sample_weight = nest.flatten(sample_weight)
        zip_args = (y_true, y_pred, sample_weight, self._metrics, self._weighted_metrics)
        for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
            if y_t is None or (all((m is None for m in metric_objs)) and all((wm is None for wm in weighted_metric_objs))):
                continue
            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            mask = get_mask(y_p)
            sw = apply_mask(y_p, sw, mask)
            for metric_obj in metric_objs:
                if metric_obj is None:
                    continue
                metric_obj.update_state(y_t, y_p, sample_weight=mask)
            for weighted_metric_obj in weighted_metric_objs:
                if weighted_metric_obj is None:
                    continue
                weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)

    def reset_state(self):
        """Resets the state of all `Metric`s in this container."""
        if self._built:
            metrics = self._metrics_in_order
        else:
            metrics = nest.flatten(self._user_metrics) + nest.flatten(self._user_weighted_metrics)
        for metric_obj in metrics:
            if isinstance(metric_obj, metrics_mod.Metric):
                metric_obj.reset_state()

    def _get_metric_objects(self, metrics, y_t, y_p):
        """Convert user-supplied metrics to `Metric` objects."""
        metrics = nest.flatten(metrics)
        return [self._get_metric_object(m, y_t, y_p) for m in metrics]

    def _get_metric_object(self, metric, y_t, y_p):
        """Converts user-supplied metric to a `Metric` object.

    Args:
      metric: A string, function, or `Metric` object.
      y_t: Sample of label.
      y_p: Sample of output.

    Returns:
      A `Metric` object.
    """
        if metric is None:
            return None
        if str(metric).lower() not in ['accuracy', 'acc', 'crossentropy', 'ce']:
            metric_obj = metrics_mod.get(metric)
        else:
            y_t_rank = len(y_t.shape.as_list())
            y_p_rank = len(y_p.shape.as_list())
            y_t_last_dim = y_t.shape.as_list()[-1]
            y_p_last_dim = y_p.shape.as_list()[-1]
            is_binary = y_p_last_dim == 1
            is_sparse_categorical = y_t_rank < y_p_rank or (y_t_last_dim == 1 and y_p_last_dim > 1)
            if str(metric).lower() in ['accuracy', 'acc']:
                if is_binary:
                    metric_obj = metrics_mod.binary_accuracy
                elif is_sparse_categorical:
                    metric_obj = metrics_mod.sparse_categorical_accuracy
                else:
                    metric_obj = metrics_mod.categorical_accuracy
            elif is_binary:
                metric_obj = metrics_mod.binary_crossentropy
            elif is_sparse_categorical:
                metric_obj = metrics_mod.sparse_categorical_crossentropy
            else:
                metric_obj = metrics_mod.categorical_crossentropy
        if isinstance(metric_obj, losses_mod.Loss):
            metric_obj._allow_sum_over_batch_size = True
        if not isinstance(metric_obj, metrics_mod.Metric):
            if isinstance(metric, str):
                metric_name = metric
            else:
                metric_name = get_custom_object_name(metric)
                if metric_name is None:
                    raise ValueError('Metric should be a callable, found: {}'.format(metric))
            metric_obj = metrics_mod.MeanMetricWrapper(metric_obj, name=metric_name)
        return metric_obj

    def _should_broadcast(self, obj):
        if not nest.is_nested(obj):
            return True
        return isinstance(obj, (list, tuple)) and (not any((nest.is_nested(o) for o in obj)))

    def _copy_object(self, obj):
        if isinstance(obj, metrics_mod.Metric):
            return obj.__class__.from_config(obj.get_config())
        return obj