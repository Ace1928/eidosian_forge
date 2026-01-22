import flatbuffers
from flatbuffers.compat import import_numpy
class TileOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tileOptions = TileOptions()
        tileOptions.Init(buf, pos)
        return cls.InitFromObj(tileOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, tileOptions):
        x = TileOptionsT()
        x._UnPack(tileOptions)
        return x

    def _UnPack(self, tileOptions):
        if tileOptions is None:
            return

    def Pack(self, builder):
        TileOptionsStart(builder)
        tileOptions = TileOptionsEnd(builder)
        return tileOptions