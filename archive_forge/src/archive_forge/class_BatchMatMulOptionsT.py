import flatbuffers
from flatbuffers.compat import import_numpy
class BatchMatMulOptionsT(object):

    def __init__(self):
        self.adjX = False
        self.adjY = False
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        batchMatMulOptions = BatchMatMulOptions()
        batchMatMulOptions.Init(buf, pos)
        return cls.InitFromObj(batchMatMulOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, batchMatMulOptions):
        x = BatchMatMulOptionsT()
        x._UnPack(batchMatMulOptions)
        return x

    def _UnPack(self, batchMatMulOptions):
        if batchMatMulOptions is None:
            return
        self.adjX = batchMatMulOptions.AdjX()
        self.adjY = batchMatMulOptions.AdjY()
        self.asymmetricQuantizeInputs = batchMatMulOptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        BatchMatMulOptionsStart(builder)
        BatchMatMulOptionsAddAdjX(builder, self.adjX)
        BatchMatMulOptionsAddAdjY(builder, self.adjY)
        BatchMatMulOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        batchMatMulOptions = BatchMatMulOptionsEnd(builder)
        return batchMatMulOptions