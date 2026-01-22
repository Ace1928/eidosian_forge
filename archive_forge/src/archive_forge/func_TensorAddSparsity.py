import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddSparsity(builder, sparsity):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(sparsity), 0)