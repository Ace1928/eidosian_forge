class GeneratedServiceType(type):
    """Metaclass for service classes created at runtime from ServiceDescriptors.

  Implementations for all methods described in the Service class are added here
  by this class. We also create properties to allow getting/setting all fields
  in the protocol message.

  The protocol compiler currently uses this metaclass to create protocol service
  classes at runtime. Clients can also manually create their own classes at
  runtime, as in this example::

    mydescriptor = ServiceDescriptor(.....)
    class MyProtoService(service.Service):
      __metaclass__ = GeneratedServiceType
      DESCRIPTOR = mydescriptor
    myservice_instance = MyProtoService()
    # ...
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'

    def __init__(cls, name, bases, dictionary):
        """Creates a message service class.

    Args:
      name: Name of the class (ignored, but required by the metaclass
        protocol).
      bases: Base classes of the class being constructed.
      dictionary: The class dictionary of the class being constructed.
        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object
        describing this protocol service type.
    """
        if GeneratedServiceType._DESCRIPTOR_KEY not in dictionary:
            return
        descriptor = dictionary[GeneratedServiceType._DESCRIPTOR_KEY]
        service_builder = _ServiceBuilder(descriptor)
        service_builder.BuildService(cls)
        cls.DESCRIPTOR = descriptor