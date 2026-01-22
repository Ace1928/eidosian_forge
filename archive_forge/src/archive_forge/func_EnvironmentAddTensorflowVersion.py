import flatbuffers
from flatbuffers.compat import import_numpy
def EnvironmentAddTensorflowVersion(builder, tensorflowVersion):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(tensorflowVersion), 0)