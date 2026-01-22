import flatbuffers
from flatbuffers.compat import import_numpy
def UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(3, timeMajor, 0)