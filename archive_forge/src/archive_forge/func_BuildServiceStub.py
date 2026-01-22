def BuildServiceStub(self, cls):
    """Constructs the stub class.

    Args:
      cls: The class that will be constructed.
    """

    def _ServiceStubInit(stub, rpc_channel):
        stub.rpc_channel = rpc_channel
    self.cls = cls
    cls.__init__ = _ServiceStubInit
    for method in self.descriptor.methods:
        setattr(cls, method.name, self._GenerateStubMethod(method))