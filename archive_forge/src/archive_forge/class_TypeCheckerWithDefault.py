import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class TypeCheckerWithDefault(TypeChecker):

    def __init__(self, default_value, *acceptable_types):
        TypeChecker.__init__(self, *acceptable_types)
        self._default_value = default_value

    def DefaultValue(self):
        return self._default_value