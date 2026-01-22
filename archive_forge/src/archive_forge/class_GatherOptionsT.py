import flatbuffers
from flatbuffers.compat import import_numpy
class GatherOptionsT(object):

    def __init__(self):
        self.axis = 0
        self.batchDims = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        gatherOptions = GatherOptions()
        gatherOptions.Init(buf, pos)
        return cls.InitFromObj(gatherOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, gatherOptions):
        x = GatherOptionsT()
        x._UnPack(gatherOptions)
        return x

    def _UnPack(self, gatherOptions):
        if gatherOptions is None:
            return
        self.axis = gatherOptions.Axis()
        self.batchDims = gatherOptions.BatchDims()

    def Pack(self, builder):
        GatherOptionsStart(builder)
        GatherOptionsAddAxis(builder, self.axis)
        GatherOptionsAddBatchDims(builder, self.batchDims)
        gatherOptions = GatherOptionsEnd(builder)
        return gatherOptions