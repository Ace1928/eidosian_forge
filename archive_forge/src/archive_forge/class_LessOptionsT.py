import flatbuffers
from flatbuffers.compat import import_numpy
class LessOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lessOptions = LessOptions()
        lessOptions.Init(buf, pos)
        return cls.InitFromObj(lessOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, lessOptions):
        x = LessOptionsT()
        x._UnPack(lessOptions)
        return x

    def _UnPack(self, lessOptions):
        if lessOptions is None:
            return

    def Pack(self, builder):
        LessOptionsStart(builder)
        lessOptions = LessOptionsEnd(builder)
        return lessOptions