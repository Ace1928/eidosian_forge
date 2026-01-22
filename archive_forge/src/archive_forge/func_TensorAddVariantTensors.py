import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddVariantTensors(builder, variantTensors):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(variantTensors), 0)