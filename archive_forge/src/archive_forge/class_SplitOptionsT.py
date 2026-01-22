import flatbuffers
from flatbuffers.compat import import_numpy
class SplitOptionsT(object):

    def __init__(self):
        self.numSplits = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        splitOptions = SplitOptions()
        splitOptions.Init(buf, pos)
        return cls.InitFromObj(splitOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, splitOptions):
        x = SplitOptionsT()
        x._UnPack(splitOptions)
        return x

    def _UnPack(self, splitOptions):
        if splitOptions is None:
            return
        self.numSplits = splitOptions.NumSplits()

    def Pack(self, builder):
        SplitOptionsStart(builder)
        SplitOptionsAddNumSplits(builder, self.numSplits)
        splitOptions = SplitOptionsEnd(builder)
        return splitOptions