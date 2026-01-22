import flatbuffers
from flatbuffers.compat import import_numpy
class PowOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        powOptions = PowOptions()
        powOptions.Init(buf, pos)
        return cls.InitFromObj(powOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, powOptions):
        x = PowOptionsT()
        x._UnPack(powOptions)
        return x

    def _UnPack(self, powOptions):
        if powOptions is None:
            return

    def Pack(self, builder):
        PowOptionsStart(builder)
        powOptions = PowOptionsEnd(builder)
        return powOptions