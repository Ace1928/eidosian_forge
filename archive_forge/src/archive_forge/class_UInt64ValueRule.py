from cloudsdk.google.protobuf import wrappers_pb2
class UInt64ValueRule(WrapperRule):
    _proto_type = wrappers_pb2.UInt64Value
    _python_type = int