import flatbuffers
from flatbuffers.compat import import_numpy
class LogicalAndOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalAndOptions = LogicalAndOptions()
        logicalAndOptions.Init(buf, pos)
        return cls.InitFromObj(logicalAndOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, logicalAndOptions):
        x = LogicalAndOptionsT()
        x._UnPack(logicalAndOptions)
        return x

    def _UnPack(self, logicalAndOptions):
        if logicalAndOptions is None:
            return

    def Pack(self, builder):
        LogicalAndOptionsStart(builder)
        logicalAndOptions = LogicalAndOptionsEnd(builder)
        return logicalAndOptions