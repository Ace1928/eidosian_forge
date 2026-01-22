class _ServiceStubBuilder(object):
    """Constructs a protocol service stub class using a service descriptor.

  Given a service descriptor, this class constructs a suitable stub class.
  A stub is just a type-safe wrapper around an RpcChannel which emulates a
  local implementation of the service.

  One service stub builder instance constructs exactly one class. It means all
  instances of that class share the same service stub builder.
  """

    def __init__(self, service_descriptor):
        """Initializes an instance of the service stub class builder.

    Args:
      service_descriptor: ServiceDescriptor to use when constructing the
        stub class.
    """
        self.descriptor = service_descriptor

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

    def _GenerateStubMethod(self, method):
        return lambda inst, rpc_controller, request, callback=None: self._StubMethod(inst, method, rpc_controller, request, callback)

    def _StubMethod(self, stub, method_descriptor, rpc_controller, request, callback):
        """The body of all service methods in the generated stub class.

    Args:
      stub: Stub instance.
      method_descriptor: Descriptor of the invoked method.
      rpc_controller: Rpc controller to execute the method.
      request: Request protocol message.
      callback: A callback to execute when the method finishes.
    Returns:
      Response message (in case of blocking call).
    """
        return stub.rpc_channel.CallMethod(method_descriptor, rpc_controller, request, method_descriptor.output_type._concrete_class, callback)