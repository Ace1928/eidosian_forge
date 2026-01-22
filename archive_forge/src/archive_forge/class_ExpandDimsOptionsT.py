import flatbuffers
from flatbuffers.compat import import_numpy
class ExpandDimsOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        expandDimsOptions = ExpandDimsOptions()
        expandDimsOptions.Init(buf, pos)
        return cls.InitFromObj(expandDimsOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, expandDimsOptions):
        x = ExpandDimsOptionsT()
        x._UnPack(expandDimsOptions)
        return x

    def _UnPack(self, expandDimsOptions):
        if expandDimsOptions is None:
            return

    def Pack(self, builder):
        ExpandDimsOptionsStart(builder)
        expandDimsOptions = ExpandDimsOptionsEnd(builder)
        return expandDimsOptions