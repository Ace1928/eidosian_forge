import flatbuffers
from flatbuffers.compat import import_numpy
class FillOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fillOptions = FillOptions()
        fillOptions.Init(buf, pos)
        return cls.InitFromObj(fillOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, fillOptions):
        x = FillOptionsT()
        x._UnPack(fillOptions)
        return x

    def _UnPack(self, fillOptions):
        if fillOptions is None:
            return

    def Pack(self, builder):
        FillOptionsStart(builder)
        fillOptions = FillOptionsEnd(builder)
        return fillOptions