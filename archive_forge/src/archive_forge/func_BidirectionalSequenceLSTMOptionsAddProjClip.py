import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceLSTMOptionsAddProjClip(builder, projClip):
    builder.PrependFloat32Slot(2, projClip, 0.0)