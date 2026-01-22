import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddBuiltinOptions(builder, builtinOptions):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(builtinOptions), 0)