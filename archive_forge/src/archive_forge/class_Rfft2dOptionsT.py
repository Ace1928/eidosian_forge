import flatbuffers
from flatbuffers.compat import import_numpy
class Rfft2dOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rfft2dOptions = Rfft2dOptions()
        rfft2dOptions.Init(buf, pos)
        return cls.InitFromObj(rfft2dOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, rfft2dOptions):
        x = Rfft2dOptionsT()
        x._UnPack(rfft2dOptions)
        return x

    def _UnPack(self, rfft2dOptions):
        if rfft2dOptions is None:
            return

    def Pack(self, builder):
        Rfft2dOptionsStart(builder)
        rfft2dOptions = Rfft2dOptionsEnd(builder)
        return rfft2dOptions