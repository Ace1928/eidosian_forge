import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def tpu_replicate_metadata(num_replicas: int, num_cores_per_replica: int=1, topology: str='', use_tpu: bool=True, device_assignment=[], computation_shape=[], host_compute_core=[], padding_map=[], step_marker_location: str='STEP_MARK_AT_ENTRY', allow_soft_placement: bool=False, use_spmd_for_xla_partitioning: bool=False, tpu_compile_options_proto: str='', name=None):
    """Metadata indicating how the TPU computation should be replicated.

  This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.

  Args:
    num_replicas: An `int` that is `>= 0`.
      Number of replicas of the computation
    num_cores_per_replica: An optional `int`. Defaults to `1`.
      Number of cores per replica. Used for model parallelism.
    topology: An optional `string`. Defaults to `""`.
      TopologyProto indicating the topology of the TPU pod slice.
    use_tpu: An optional `bool`. Defaults to `True`.
      Whether to place the computation on the TPU.
    device_assignment: An optional list of `ints`. Defaults to `[]`.
      The assignment of devices for the computation.
    computation_shape: An optional list of `ints`. Defaults to `[]`.
      DEPRECATED. Use num_cores_per_replica instead.
    host_compute_core: An optional list of `strings`. Defaults to `[]`.
    padding_map: An optional list of `strings`. Defaults to `[]`.
    step_marker_location: An optional `string`. Defaults to `"STEP_MARK_AT_ENTRY"`.
    allow_soft_placement: An optional `bool`. Defaults to `False`.
    use_spmd_for_xla_partitioning: An optional `bool`. Defaults to `False`.
    tpu_compile_options_proto: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUReplicateMetadata', name, 'num_replicas', num_replicas, 'num_cores_per_replica', num_cores_per_replica, 'topology', topology, 'use_tpu', use_tpu, 'device_assignment', device_assignment, 'computation_shape', computation_shape, 'host_compute_core', host_compute_core, 'padding_map', padding_map, 'step_marker_location', step_marker_location, 'allow_soft_placement', allow_soft_placement, 'use_spmd_for_xla_partitioning', use_spmd_for_xla_partitioning, 'tpu_compile_options_proto', tpu_compile_options_proto)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_replicate_metadata_eager_fallback(num_replicas=num_replicas, num_cores_per_replica=num_cores_per_replica, topology=topology, use_tpu=use_tpu, device_assignment=device_assignment, computation_shape=computation_shape, host_compute_core=host_compute_core, padding_map=padding_map, step_marker_location=step_marker_location, allow_soft_placement=allow_soft_placement, use_spmd_for_xla_partitioning=use_spmd_for_xla_partitioning, tpu_compile_options_proto=tpu_compile_options_proto, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_replicas = _execute.make_int(num_replicas, 'num_replicas')
    if num_cores_per_replica is None:
        num_cores_per_replica = 1
    num_cores_per_replica = _execute.make_int(num_cores_per_replica, 'num_cores_per_replica')
    if topology is None:
        topology = ''
    topology = _execute.make_str(topology, 'topology')
    if use_tpu is None:
        use_tpu = True
    use_tpu = _execute.make_bool(use_tpu, 'use_tpu')
    if device_assignment is None:
        device_assignment = []
    if not isinstance(device_assignment, (list, tuple)):
        raise TypeError("Expected list for 'device_assignment' argument to 'tpu_replicate_metadata' Op, not %r." % device_assignment)
    device_assignment = [_execute.make_int(_i, 'device_assignment') for _i in device_assignment]
    if computation_shape is None:
        computation_shape = []
    if not isinstance(computation_shape, (list, tuple)):
        raise TypeError("Expected list for 'computation_shape' argument to 'tpu_replicate_metadata' Op, not %r." % computation_shape)
    computation_shape = [_execute.make_int(_i, 'computation_shape') for _i in computation_shape]
    if host_compute_core is None:
        host_compute_core = []
    if not isinstance(host_compute_core, (list, tuple)):
        raise TypeError("Expected list for 'host_compute_core' argument to 'tpu_replicate_metadata' Op, not %r." % host_compute_core)
    host_compute_core = [_execute.make_str(_s, 'host_compute_core') for _s in host_compute_core]
    if padding_map is None:
        padding_map = []
    if not isinstance(padding_map, (list, tuple)):
        raise TypeError("Expected list for 'padding_map' argument to 'tpu_replicate_metadata' Op, not %r." % padding_map)
    padding_map = [_execute.make_str(_s, 'padding_map') for _s in padding_map]
    if step_marker_location is None:
        step_marker_location = 'STEP_MARK_AT_ENTRY'
    step_marker_location = _execute.make_str(step_marker_location, 'step_marker_location')
    if allow_soft_placement is None:
        allow_soft_placement = False
    allow_soft_placement = _execute.make_bool(allow_soft_placement, 'allow_soft_placement')
    if use_spmd_for_xla_partitioning is None:
        use_spmd_for_xla_partitioning = False
    use_spmd_for_xla_partitioning = _execute.make_bool(use_spmd_for_xla_partitioning, 'use_spmd_for_xla_partitioning')
    if tpu_compile_options_proto is None:
        tpu_compile_options_proto = ''
    tpu_compile_options_proto = _execute.make_str(tpu_compile_options_proto, 'tpu_compile_options_proto')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUReplicateMetadata', num_replicas=num_replicas, num_cores_per_replica=num_cores_per_replica, topology=topology, use_tpu=use_tpu, device_assignment=device_assignment, computation_shape=computation_shape, host_compute_core=host_compute_core, padding_map=padding_map, step_marker_location=step_marker_location, allow_soft_placement=allow_soft_placement, use_spmd_for_xla_partitioning=use_spmd_for_xla_partitioning, tpu_compile_options_proto=tpu_compile_options_proto, name=name)
    return _op