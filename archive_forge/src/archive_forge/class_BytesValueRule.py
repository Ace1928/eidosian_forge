from cloudsdk.google.protobuf import wrappers_pb2
class BytesValueRule(WrapperRule):
    _proto_type = wrappers_pb2.BytesValue
    _python_type = bytes