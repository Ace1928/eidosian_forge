import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def InitFromObj(cls, model):
    x = ModelT()
    x._UnPack(model)
    return x