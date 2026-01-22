import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.TrackableDataStructure', v1=[])
class TrackableDataStructure(base.Trackable):
    """Base class for data structures which contain trackable objects."""

    def __init__(self):
        self._self_trainable = True
        self._self_extra_variables = []
        self._self_attribute_sentinel = layer_utils.AttributeSentinel(True)

    @property
    def _attribute_sentinel(self):
        return self._self_attribute_sentinel

    @property
    def trainable(self):
        return self._self_trainable

    @trainable.setter
    def trainable(self, value):
        self._self_trainable = value

    def _track_value(self, value, name):
        """Add a dependency on `value`."""
        value = sticky_attribute_assignment(trackable=self, value=value, name=name)
        if isinstance(value, variables.Variable):
            self._self_extra_variables.append(value)
        if not isinstance(value, base.Trackable):
            raise _UntrackableError(value)
        if hasattr(value, '_use_resource_variables'):
            value._use_resource_variables = True
        value_attribute_sentinel = getattr(value, '_attribute_sentinel', None)
        if value_attribute_sentinel:
            value_attribute_sentinel.add_parent(self._attribute_sentinel)
        return value

    @property
    def _values(self):
        """An iterable/sequence which may contain trackable objects."""
        raise NotImplementedError('Abstract method')

    @property
    def _layers(self):
        """All Layers and Layer containers, including empty containers."""
        collected = []
        for obj in self._values:
            if isinstance(obj, TrackableDataStructure) or layer_utils.is_layer(obj) or layer_utils.has_weights(obj):
                collected.append(obj)
        return collected

    @property
    def layers(self):
        return list(layer_utils.filter_empty_layer_containers(self._layers))

    @property
    def trainable_weights(self):
        if not self._self_trainable:
            return []
        trainable_variables = []
        for obj in self._values:
            if isinstance(obj, base.Trackable) and hasattr(obj, 'trainable_variables'):
                trainable_variables += obj.trainable_variables
        trainable_extra_variables = [v for v in self._self_extra_variables if v.trainable]
        return trainable_variables + trainable_extra_variables

    @property
    def non_trainable_weights(self):
        trainable_extra_variables = [v for v in self._self_extra_variables if v.trainable]
        non_trainable_extra_variables = [v for v in self._self_extra_variables if not v.trainable]
        non_trainable_variables = []
        for obj in self._values:
            if isinstance(obj, base.Trackable) and hasattr(obj, 'non_trainable_variables'):
                non_trainable_variables += obj.non_trainable_variables
        if not self._self_trainable:
            trainable_variables = []
            for obj in self._values:
                if isinstance(obj, base.Trackable) and hasattr(obj, 'trainable_variables'):
                    trainable_variables += obj.trainable_variables
            non_trainable_variables = trainable_variables + trainable_extra_variables + non_trainable_variables + non_trainable_extra_variables
        else:
            non_trainable_variables = non_trainable_variables + non_trainable_extra_variables
        return non_trainable_variables

    @property
    def weights(self):
        return self.trainable_weights + self.non_trainable_weights

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @property
    def variables(self):
        return self.weights

    @property
    def updates(self):
        """Aggregate updates from any `Layer` instances."""
        aggregated = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                aggregated += layer.updates
        return aggregated

    @property
    def losses(self):
        """Aggregate losses from any `Layer` instances."""
        aggregated = []
        for layer in self.layers:
            if hasattr(layer, 'losses'):
                aggregated += layer.losses
        return aggregated

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other