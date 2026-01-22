import flatbuffers
from flatbuffers.compat import import_numpy
class SquareOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squareOptions = SquareOptions()
        squareOptions.Init(buf, pos)
        return cls.InitFromObj(squareOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, squareOptions):
        x = SquareOptionsT()
        x._UnPack(squareOptions)
        return x

    def _UnPack(self, squareOptions):
        if squareOptions is None:
            return

    def Pack(self, builder):
        SquareOptionsStart(builder)
        squareOptions = SquareOptionsEnd(builder)
        return squareOptions