import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def InitFromBuf(cls, buf, pos):
    model = Model()
    model.Init(buf, pos)
    return cls.InitFromObj(model)