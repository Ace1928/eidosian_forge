import flatbuffers
from flatbuffers.compat import import_numpy
class LogicalNotOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalNotOptions = LogicalNotOptions()
        logicalNotOptions.Init(buf, pos)
        return cls.InitFromObj(logicalNotOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, logicalNotOptions):
        x = LogicalNotOptionsT()
        x._UnPack(logicalNotOptions)
        return x

    def _UnPack(self, logicalNotOptions):
        if logicalNotOptions is None:
            return

    def Pack(self, builder):
        LogicalNotOptionsStart(builder)
        logicalNotOptions = LogicalNotOptionsEnd(builder)
        return logicalNotOptions