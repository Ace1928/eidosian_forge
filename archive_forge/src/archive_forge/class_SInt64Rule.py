from proto.primitives import ProtoType
class SInt64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.SINT64