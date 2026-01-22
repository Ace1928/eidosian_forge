import flatbuffers
from flatbuffers.compat import import_numpy
class DequantizeOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dequantizeOptions = DequantizeOptions()
        dequantizeOptions.Init(buf, pos)
        return cls.InitFromObj(dequantizeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, dequantizeOptions):
        x = DequantizeOptionsT()
        x._UnPack(dequantizeOptions)
        return x

    def _UnPack(self, dequantizeOptions):
        if dequantizeOptions is None:
            return

    def Pack(self, builder):
        DequantizeOptionsStart(builder)
        dequantizeOptions = DequantizeOptionsEnd(builder)
        return dequantizeOptions