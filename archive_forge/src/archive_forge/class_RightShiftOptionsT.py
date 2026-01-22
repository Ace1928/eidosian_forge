import flatbuffers
from flatbuffers.compat import import_numpy
class RightShiftOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rightShiftOptions = RightShiftOptions()
        rightShiftOptions.Init(buf, pos)
        return cls.InitFromObj(rightShiftOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, rightShiftOptions):
        x = RightShiftOptionsT()
        x._UnPack(rightShiftOptions)
        return x

    def _UnPack(self, rightShiftOptions):
        if rightShiftOptions is None:
            return

    def Pack(self, builder):
        RightShiftOptionsStart(builder)
        rightShiftOptions = RightShiftOptionsEnd(builder)
        return rightShiftOptions