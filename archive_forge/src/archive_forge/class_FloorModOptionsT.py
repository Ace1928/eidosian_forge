import flatbuffers
from flatbuffers.compat import import_numpy
class FloorModOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        floorModOptions = FloorModOptions()
        floorModOptions.Init(buf, pos)
        return cls.InitFromObj(floorModOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, floorModOptions):
        x = FloorModOptionsT()
        x._UnPack(floorModOptions)
        return x

    def _UnPack(self, floorModOptions):
        if floorModOptions is None:
            return

    def Pack(self, builder):
        FloorModOptionsStart(builder)
        floorModOptions = FloorModOptionsEnd(builder)
        return floorModOptions