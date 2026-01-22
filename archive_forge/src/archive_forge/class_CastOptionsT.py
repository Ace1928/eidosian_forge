import flatbuffers
from flatbuffers.compat import import_numpy
class CastOptionsT(object):

    def __init__(self):
        self.inDataType = 0
        self.outDataType = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        castOptions = CastOptions()
        castOptions.Init(buf, pos)
        return cls.InitFromObj(castOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, castOptions):
        x = CastOptionsT()
        x._UnPack(castOptions)
        return x

    def _UnPack(self, castOptions):
        if castOptions is None:
            return
        self.inDataType = castOptions.InDataType()
        self.outDataType = castOptions.OutDataType()

    def Pack(self, builder):
        CastOptionsStart(builder)
        CastOptionsAddInDataType(builder, self.inDataType)
        CastOptionsAddOutDataType(builder, self.outDataType)
        castOptions = CastOptionsEnd(builder)
        return castOptions