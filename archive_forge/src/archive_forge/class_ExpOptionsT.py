import flatbuffers
from flatbuffers.compat import import_numpy
class ExpOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        expOptions = ExpOptions()
        expOptions.Init(buf, pos)
        return cls.InitFromObj(expOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, expOptions):
        x = ExpOptionsT()
        x._UnPack(expOptions)
        return x

    def _UnPack(self, expOptions):
        if expOptions is None:
            return

    def Pack(self, builder):
        ExpOptionsStart(builder)
        expOptions = ExpOptionsEnd(builder)
        return expOptions