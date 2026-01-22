from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for _MutableDenseHashTable."""

    def __init__(self, table, name):
        tensors = table.export()
        specs = [BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
        super(_MutableDenseHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
        del restored_shapes
        with ops.colocate_with(self.op.resource_handle):
            return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle, restored_tensors[0], restored_tensors[1])