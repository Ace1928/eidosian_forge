import flatbuffers
from flatbuffers.compat import import_numpy
class QuantizeOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quantizeOptions = QuantizeOptions()
        quantizeOptions.Init(buf, pos)
        return cls.InitFromObj(quantizeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, quantizeOptions):
        x = QuantizeOptionsT()
        x._UnPack(quantizeOptions)
        return x

    def _UnPack(self, quantizeOptions):
        if quantizeOptions is None:
            return

    def Pack(self, builder):
        QuantizeOptionsStart(builder)
        quantizeOptions = QuantizeOptionsEnd(builder)
        return quantizeOptions