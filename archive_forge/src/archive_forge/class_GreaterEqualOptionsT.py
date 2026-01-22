import flatbuffers
from flatbuffers.compat import import_numpy
class GreaterEqualOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        greaterEqualOptions = GreaterEqualOptions()
        greaterEqualOptions.Init(buf, pos)
        return cls.InitFromObj(greaterEqualOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, greaterEqualOptions):
        x = GreaterEqualOptionsT()
        x._UnPack(greaterEqualOptions)
        return x

    def _UnPack(self, greaterEqualOptions):
        if greaterEqualOptions is None:
            return

    def Pack(self, builder):
        GreaterEqualOptionsStart(builder)
        greaterEqualOptions = GreaterEqualOptionsEnd(builder)
        return greaterEqualOptions