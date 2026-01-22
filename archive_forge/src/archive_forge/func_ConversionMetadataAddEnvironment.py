import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionMetadataAddEnvironment(builder, environment):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(environment), 0)