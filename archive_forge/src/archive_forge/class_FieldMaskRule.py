from cloudsdk.google.protobuf import field_mask_pb2
class FieldMaskRule:
    """A marshal between FieldMask and strings.

    See https://github.com/googleapis/proto-plus-python/issues/333
    and
    https://developers.google.com/protocol-buffers/docs/proto3#json
    for more details.
    """

    def to_python(self, value, *, absent: bool=None):
        return value

    def to_proto(self, value):
        if isinstance(value, str):
            field_mask_value = field_mask_pb2.FieldMask()
            field_mask_value.FromJsonString(value=value)
            return field_mask_value
        return value