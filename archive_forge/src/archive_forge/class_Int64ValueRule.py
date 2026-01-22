from cloudsdk.google.protobuf import wrappers_pb2
class Int64ValueRule(WrapperRule):
    _proto_type = wrappers_pb2.Int64Value
    _python_type = int