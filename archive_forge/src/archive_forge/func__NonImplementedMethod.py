def _NonImplementedMethod(self, method_name, rpc_controller, callback):
    """The body of all methods in the generated service class.

    Args:
      method_name: Name of the method being executed.
      rpc_controller: RPC controller used to execute this method.
      callback: A callback which will be invoked when the method finishes.
    """
    rpc_controller.SetFailed('Method %s not implemented.' % method_name)
    callback(None)