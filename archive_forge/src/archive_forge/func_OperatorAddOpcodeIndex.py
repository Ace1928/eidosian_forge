import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddOpcodeIndex(builder, opcodeIndex):
    builder.PrependUint32Slot(0, opcodeIndex, 0)