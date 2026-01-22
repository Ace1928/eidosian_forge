from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
def GetPrototype(self, descriptor):
    """Obtains a proto2 message class based on the passed in descriptor.

    Passing a descriptor with a fully qualified name matching a previous
    invocation will cause the same class to be returned.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
    if descriptor not in self._classes:
        result_class = self.CreatePrototype(descriptor)
        self._classes[descriptor] = result_class
        return result_class
    return self._classes[descriptor]