import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddSignatureDefs(builder, signatureDefs):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(signatureDefs), 0)