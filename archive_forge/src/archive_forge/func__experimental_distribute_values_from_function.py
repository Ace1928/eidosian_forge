import functools
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _experimental_distribute_values_from_function(self, value_fn):
    per_replica_values = []
    for i in range(self._mesh.num_local_devices()):
        replica_id = d_config.client_id() * self._mesh.num_local_devices() + i
        per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
    result = distribute_utils.regroup(per_replica_values, always_wrap=True)
    map_fn = functools.partial(dtensor_util.convert_per_replica_to_dtensor, mesh=self._mesh)
    return nest.map_structure(map_fn, result)