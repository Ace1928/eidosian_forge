import flatbuffers
from flatbuffers.compat import import_numpy
class AbsOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        absOptions = AbsOptions()
        absOptions.Init(buf, pos)
        return cls.InitFromObj(absOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, absOptions):
        x = AbsOptionsT()
        x._UnPack(absOptions)
        return x

    def _UnPack(self, absOptions):
        if absOptions is None:
            return

    def Pack(self, builder):
        AbsOptionsStart(builder)
        absOptions = AbsOptionsEnd(builder)
        return absOptions