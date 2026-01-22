import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddIsVariable(builder, isVariable):
    builder.PrependBoolSlot(5, isVariable, 0)