import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddBuiltinOptionsType(builder, builtinOptionsType):
    builder.PrependUint8Slot(3, builtinOptionsType, 0)