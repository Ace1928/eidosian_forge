import flatbuffers
from flatbuffers.compat import import_numpy
class NegOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        negOptions = NegOptions()
        negOptions.Init(buf, pos)
        return cls.InitFromObj(negOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, negOptions):
        x = NegOptionsT()
        x._UnPack(negOptions)
        return x

    def _UnPack(self, negOptions):
        if negOptions is None:
            return

    def Pack(self, builder):
        NegOptionsStart(builder)
        negOptions = NegOptionsEnd(builder)
        return negOptions