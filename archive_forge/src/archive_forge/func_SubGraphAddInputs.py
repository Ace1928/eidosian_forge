import flatbuffers
from flatbuffers.compat import import_numpy
def SubGraphAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)