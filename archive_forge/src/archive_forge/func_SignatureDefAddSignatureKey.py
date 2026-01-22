import flatbuffers
from flatbuffers.compat import import_numpy
def SignatureDefAddSignatureKey(builder, signatureKey):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(signatureKey), 0)