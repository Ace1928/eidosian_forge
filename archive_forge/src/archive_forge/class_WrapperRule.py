from cloudsdk.google.protobuf import wrappers_pb2
class WrapperRule:
    """A marshal for converting the protobuf wrapper classes to Python.

    This class converts between ``google.protobuf.BoolValue``,
    ``google.protobuf.StringValue``, and their siblings to the appropriate
    Python equivalents.

    These are effectively similar to the protobuf primitives except
    that None becomes a possible value.
    """

    def to_python(self, value, *, absent: bool=None):
        if isinstance(value, self._proto_type):
            if absent:
                return None
            return value.value
        return value

    def to_proto(self, value):
        if isinstance(value, self._python_type):
            return self._proto_type(value=value)
        return value