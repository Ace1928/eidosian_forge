import abc
import threading
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _IteratorSaveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject for saving/restoring iterator state."""

    def __init__(self, iterator_resource, name, external_state_policy=options_lib.ExternalStatePolicy.FAIL):
        serialized_iterator = gen_dataset_ops.serialize_iterator(iterator_resource, external_state_policy=external_state_policy.value)
        specs = [BaseSaverBuilder.SaveSpec(serialized_iterator, '', name + '_STATE', device=iterator_resource.device)]
        super(_IteratorSaveable, self).__init__(iterator_resource, specs, name)

    def restore(self, restored_tensors, restored_shapes):
        with ops.colocate_with(self.op):
            return gen_dataset_ops.deserialize_iterator(self.op, restored_tensors[0])