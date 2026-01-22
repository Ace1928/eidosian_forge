from typing import Optional, Sequence, Union
import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class GrpcClient(Client):
    """Client wrapper to connect to remote RPC server using GRPC.

  If Client is created with (list_registered_methods=True):
  1. Input and output specs for the methods till this point will be fetched from
  Server.
  2. convenience methods are added to invoke registered methods directly from
  client.
  For example:
    For call a server method `add`
    client.add(a, b) or client.add_async(a, b) can be used instead of
    client.call(args=[a,b], output_specs=[..])

  Prerequiste for using list_registered_methods=True:
   1. Server should be already started with the registered methods.
   2. Client must be created in Eager mode.
  """

    def __init__(self, address: str, name: str='', list_registered_methods=False, timeout_in_ms=0):
        self._client_handle, methods = gen_rpc_ops.rpc_client(shared_name=name, server_address=address, list_registered_methods=list_registered_methods, timeout_in_ms=timeout_in_ms)
        if context.executing_eagerly():
            self._handle_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._client_handle, handle_device=self._client_handle.device)
        else:
            raise NotImplementedError('Client creation is supported only in eager mode.')
        self._server_address = address
        self._method_registry = {}
        for method in methods.numpy():
            m = rpc_pb2.RegisteredMethod()
            m.ParseFromString(method)
            output_specs = nested_structure_coder.decode_proto(m.output_specs)
            input_specs = nested_structure_coder.decode_proto(m.input_specs)
            self._method_registry[m.method] = output_specs
            doc_string = 'RPC Call for ' + m.method + ' method to server ' + address
            self._add_method(m.method, output_specs, input_specs, self._client_handle, doc_string)

    def _add_method(self, method_name, output_specs, input_specs, client_handle, doc_string):
        """Method to add RPC methods to the client object."""

        def validate_and_get_flat_inputs(*args):
            if args is None:
                args = []
            if input_specs:
                nest.assert_same_structure(args, input_specs)
            flat_inputs = nest.flatten(args)
            return flat_inputs

        def call_wrapper(*args, timeout_in_ms=0):
            status_or, deleter = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
            return StatusOrResult(status_or, deleter, output_specs)

        def call_blocking_wrapper(*args, timeout_in_ms=0):
            status_or, deleter = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
            status_or = StatusOrResult(status_or, deleter, output_specs)
            if status_or.is_ok():
                return status_or.get_value()
            else:
                error_code, error_msg = status_or.get_error()
                raise errors.exception_type_from_error_code(error_code.numpy())(None, None, error_msg.numpy())
        setattr(self, method_name, call_wrapper)
        call_wrapper.__doc__ = doc_string
        blocking_method_name = method_name + '_blocking'
        setattr(self, blocking_method_name, call_blocking_wrapper)
        call_blocking_wrapper.__doc__ = doc_string

    def call(self, method_name: str, args: Optional[Sequence[core_tf_types.Tensor]]=None, output_specs=None, timeout_in_ms=0):
        """Method to invoke remote registered functions on the connected server.

    Server should be started before making an RPC Call.

    Args:
      method_name: Registered method to invoke on Server.
      args: Input arguments for the method.
      output_specs: Output specs for the output from method.
      timeout_in_ms: Timeout for this call. If 0, default client timeout will be
       used.

    Returns:
      StatusOrResult object. This function issues the RPC call to server, it
      does not block for the duration of RPC. Please call is_ok, get_error or
      get_value methods on the returned object to blocked till RPC finishes.
    """
        if args is None:
            args = []
        status_or, deleter = gen_rpc_ops.rpc_call(self._client_handle, args=nest.flatten(args), method_name=method_name, timeout_in_ms=timeout_in_ms)
        return StatusOrResult(status_or, deleter, output_specs)