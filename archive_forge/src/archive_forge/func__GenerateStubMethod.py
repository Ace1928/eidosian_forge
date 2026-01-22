def _GenerateStubMethod(self, method):
    return lambda inst, rpc_controller, request, callback=None: self._StubMethod(inst, method, rpc_controller, request, callback)