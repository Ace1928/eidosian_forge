import flatbuffers
from flatbuffers.compat import import_numpy
class Conv3DOptionsT(object):

    def __init__(self):
        self.padding = 0
        self.strideD = 0
        self.strideW = 0
        self.strideH = 0
        self.fusedActivationFunction = 0
        self.dilationDFactor = 1
        self.dilationWFactor = 1
        self.dilationHFactor = 1

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv3Doptions = Conv3DOptions()
        conv3Doptions.Init(buf, pos)
        return cls.InitFromObj(conv3Doptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, conv3Doptions):
        x = Conv3DOptionsT()
        x._UnPack(conv3Doptions)
        return x

    def _UnPack(self, conv3Doptions):
        if conv3Doptions is None:
            return
        self.padding = conv3Doptions.Padding()
        self.strideD = conv3Doptions.StrideD()
        self.strideW = conv3Doptions.StrideW()
        self.strideH = conv3Doptions.StrideH()
        self.fusedActivationFunction = conv3Doptions.FusedActivationFunction()
        self.dilationDFactor = conv3Doptions.DilationDFactor()
        self.dilationWFactor = conv3Doptions.DilationWFactor()
        self.dilationHFactor = conv3Doptions.DilationHFactor()

    def Pack(self, builder):
        Conv3DOptionsStart(builder)
        Conv3DOptionsAddPadding(builder, self.padding)
        Conv3DOptionsAddStrideD(builder, self.strideD)
        Conv3DOptionsAddStrideW(builder, self.strideW)
        Conv3DOptionsAddStrideH(builder, self.strideH)
        Conv3DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        Conv3DOptionsAddDilationDFactor(builder, self.dilationDFactor)
        Conv3DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        Conv3DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        conv3Doptions = Conv3DOptionsEnd(builder)
        return conv3Doptions