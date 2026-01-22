from google.protobuf.internal import api_implementation
def InternalAdd(field_number, wire_type, data):
    unknown_field = UnknownField(field_number, wire_type, data)
    self._values.append(unknown_field)