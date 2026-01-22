from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
def CreatePrototype(self, descriptor):
    """Builds a proto2 message class based on the passed in descriptor.

    Don't call this function directly, it always creates a new class. Call
    GetPrototype() instead. This method is meant to be overridden in subblasses
    to perform additional operations on the newly constructed class.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
    descriptor_name = descriptor.name
    result_class = _GENERATED_PROTOCOL_MESSAGE_TYPE(descriptor_name, (message.Message,), {'DESCRIPTOR': descriptor, '__module__': None})
    result_class._FACTORY = self
    self._classes[descriptor] = result_class
    for field in descriptor.fields:
        if field.message_type:
            self.GetPrototype(field.message_type)
    for extension in result_class.DESCRIPTOR.extensions:
        if extension.containing_type not in self._classes:
            self.GetPrototype(extension.containing_type)
        extended_class = self._classes[extension.containing_type]
        extended_class.RegisterExtension(extension)
    return result_class