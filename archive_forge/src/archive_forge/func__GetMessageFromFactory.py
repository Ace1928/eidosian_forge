from collections import OrderedDict
import hashlib
import os
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message_factory
def _GetMessageFromFactory(factory, full_name):
    """Get a proto class from the MessageFactory by name.

  Args:
    factory: a MessageFactory instance.
    full_name: str, the fully qualified name of the proto type.
  Returns:
    A class, for the type identified by full_name.
  Raises:
    KeyError, if the proto is not found in the factory's descriptor pool.
  """
    proto_descriptor = factory.pool.FindMessageTypeByName(full_name)
    proto_cls = factory.GetPrototype(proto_descriptor)
    return proto_cls