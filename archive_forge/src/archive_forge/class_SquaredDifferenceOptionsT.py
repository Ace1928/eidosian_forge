import flatbuffers
from flatbuffers.compat import import_numpy
class SquaredDifferenceOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squaredDifferenceOptions = SquaredDifferenceOptions()
        squaredDifferenceOptions.Init(buf, pos)
        return cls.InitFromObj(squaredDifferenceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, squaredDifferenceOptions):
        x = SquaredDifferenceOptionsT()
        x._UnPack(squaredDifferenceOptions)
        return x

    def _UnPack(self, squaredDifferenceOptions):
        if squaredDifferenceOptions is None:
            return

    def Pack(self, builder):
        SquaredDifferenceOptionsStart(builder)
        squaredDifferenceOptions = SquaredDifferenceOptionsEnd(builder)
        return squaredDifferenceOptions