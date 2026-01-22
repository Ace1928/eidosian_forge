def _GenerateNonImplementedMethod(self, method):
    """Generates and returns a method that can be set for a service methods.

    Args:
      method: Descriptor of the service method for which a method is to be
        generated.

    Returns:
      A method that can be added to the service class.
    """
    return lambda inst, rpc_controller, request, callback: self._NonImplementedMethod(method.name, rpc_controller, callback)