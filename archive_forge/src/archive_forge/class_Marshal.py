import abc
import enum
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import struct_pb2
from cloudsdk.google.protobuf import wrappers_pb2
from proto.marshal import compat
from proto.marshal.collections import MapComposite
from proto.marshal.collections import Repeated
from proto.marshal.collections import RepeatedComposite
from proto.marshal.rules import bytes as pb_bytes
from proto.marshal.rules import stringy_numbers
from proto.marshal.rules import dates
from proto.marshal.rules import struct
from proto.marshal.rules import wrappers
from proto.marshal.rules import field_mask
from proto.primitives import ProtoType
class Marshal(BaseMarshal):
    """The translator between protocol buffer and Python instances.

    The bulk of the implementation is in :class:`BaseMarshal`. This class
    adds identity tracking: multiple instantiations of :class:`Marshal` with
    the same name will provide the same instance.
    """
    _instances = {}

    def __new__(cls, *, name: str):
        """Create a marshal instance.

        Args:
            name (str): The name of the marshal. Instantiating multiple
                marshals with the same ``name`` argument will provide the
                same marshal each time.
        """
        klass = cls._instances.get(name)
        if klass is None:
            klass = cls._instances[name] = super().__new__(cls)
        return klass

    def __init__(self, *, name: str):
        """Instantiate a marshal.

        Args:
            name (str): The name of the marshal. Instantiating multiple
                marshals with the same ``name`` argument will provide the
                same marshal each time.
        """
        self._name = name
        if not hasattr(self, '_rules'):
            super().__init__()