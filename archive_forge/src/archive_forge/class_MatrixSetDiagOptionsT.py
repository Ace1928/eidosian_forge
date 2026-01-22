import flatbuffers
from flatbuffers.compat import import_numpy
class MatrixSetDiagOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        matrixSetDiagOptions = MatrixSetDiagOptions()
        matrixSetDiagOptions.Init(buf, pos)
        return cls.InitFromObj(matrixSetDiagOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, matrixSetDiagOptions):
        x = MatrixSetDiagOptionsT()
        x._UnPack(matrixSetDiagOptions)
        return x

    def _UnPack(self, matrixSetDiagOptions):
        if matrixSetDiagOptions is None:
            return

    def Pack(self, builder):
        MatrixSetDiagOptionsStart(builder)
        matrixSetDiagOptions = MatrixSetDiagOptionsEnd(builder)
        return matrixSetDiagOptions