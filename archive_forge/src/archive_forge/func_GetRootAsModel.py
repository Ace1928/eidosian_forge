import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def GetRootAsModel(cls, buf, offset=0):
    """This method is deprecated. Please switch to GetRootAs."""
    return cls.GetRootAs(buf, offset)