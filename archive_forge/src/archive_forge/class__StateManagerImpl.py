import abc
import collections
import math
import re
import numpy as np
import six
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('__internal__.feature_column.StateManager', v1=[])
class _StateManagerImpl(StateManager):
    """Manages the state of DenseFeatures and LinearLayer.

  Some `FeatureColumn`s create variables or resources to assist their
  computation. The `StateManager` is responsible for creating and storing these
  objects since `FeatureColumn`s are supposed to be stateless configuration
  only.
  """

    def __init__(self, layer, trainable):
        """Creates an _StateManagerImpl object.

    Args:
      layer: The input layer this state manager is associated with.
      trainable: Whether by default, variables created are trainable or not.
    """
        self._trainable = trainable
        self._layer = layer
        if self._layer is not None and (not hasattr(self._layer, '_resources')):
            self._layer._resources = data_structures.Mapping()
        self._cols_to_vars_map = collections.defaultdict(lambda: {})
        self._cols_to_resources_map = collections.defaultdict(lambda: {})

    def create_variable(self, feature_column, name, shape, dtype=None, trainable=True, use_resource=True, initializer=None):
        """Creates a new variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      trainable: Whether this variable is trainable or not.
      use_resource: If true, we use resource variables. Otherwise we use
        RefVariable.
      initializer: initializer instance (callable).

    Returns:
      The created variable.
    """
        if name in self._cols_to_vars_map[feature_column]:
            raise ValueError('Variable already exists.')
        with trackable.no_manual_dependency_tracking_scope(self._layer):
            var = self._layer.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=self._trainable and trainable, use_resource=use_resource, getter=variable_scope.get_variable)
        if isinstance(var, variables.PartitionedVariable):
            for v in var:
                part_name = name + '/' + str(v._get_save_slice_info().var_offset[0])
                self._layer._track_trackable(v, feature_column.name + '/' + part_name)
        elif isinstance(var, trackable.Trackable):
            self._layer._track_trackable(var, feature_column.name + '/' + name)
        self._cols_to_vars_map[feature_column][name] = var
        return var

    def get_variable(self, feature_column, name):
        """Returns an existing variable.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      name: variable name.
    """
        if name in self._cols_to_vars_map[feature_column]:
            return self._cols_to_vars_map[feature_column][name]
        raise ValueError('Variable does not exist.')

    def add_resource(self, feature_column, resource_name, resource):
        """Creates a new resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this resource corresponds to.
      resource_name: Name of the resource.
      resource: The resource.

    Returns:
      The created resource.
    """
        self._cols_to_resources_map[feature_column][resource_name] = resource
        if self._layer is not None and isinstance(resource, trackable.Trackable):
            if feature_column.name not in self._layer._resources:
                self._layer._resources[feature_column.name] = data_structures.Mapping()
            if resource_name not in self._layer._resources[feature_column.name]:
                self._layer._resources[feature_column.name][resource_name] = resource

    def has_resource(self, feature_column, resource_name):
        """Returns true iff a resource with same name exists.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      resource_name: Name of the resource.
    """
        return resource_name in self._cols_to_resources_map[feature_column]

    def get_resource(self, feature_column, resource_name):
        """Returns an already created resource.

    Resources can be things such as tables, variables, trackables, etc.

    Args:
      feature_column: A `FeatureColumn` object this variable corresponds to.
      resource_name: Name of the resource.
    """
        if feature_column not in self._cols_to_resources_map or resource_name not in self._cols_to_resources_map[feature_column]:
            raise ValueError('Resource does not exist.')
        return self._cols_to_resources_map[feature_column][resource_name]