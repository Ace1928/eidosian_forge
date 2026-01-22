import flatbuffers
from flatbuffers.compat import import_numpy
class HardSwishOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hardSwishOptions = HardSwishOptions()
        hardSwishOptions.Init(buf, pos)
        return cls.InitFromObj(hardSwishOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, hardSwishOptions):
        x = HardSwishOptionsT()
        x._UnPack(hardSwishOptions)
        return x

    def _UnPack(self, hardSwishOptions):
        if hardSwishOptions is None:
            return

    def Pack(self, builder):
        HardSwishOptionsStart(builder)
        hardSwishOptions = HardSwishOptionsEnd(builder)
        return hardSwishOptions