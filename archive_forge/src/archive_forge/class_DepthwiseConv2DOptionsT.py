import flatbuffers
from flatbuffers.compat import import_numpy
class DepthwiseConv2DOptionsT(object):

    def __init__(self):
        self.padding = 0
        self.strideW = 0
        self.strideH = 0
        self.depthMultiplier = 0
        self.fusedActivationFunction = 0
        self.dilationWFactor = 1
        self.dilationHFactor = 1

    @classmethod
    def InitFromBuf(cls, buf, pos):
        depthwiseConv2Doptions = DepthwiseConv2DOptions()
        depthwiseConv2Doptions.Init(buf, pos)
        return cls.InitFromObj(depthwiseConv2Doptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, depthwiseConv2Doptions):
        x = DepthwiseConv2DOptionsT()
        x._UnPack(depthwiseConv2Doptions)
        return x

    def _UnPack(self, depthwiseConv2Doptions):
        if depthwiseConv2Doptions is None:
            return
        self.padding = depthwiseConv2Doptions.Padding()
        self.strideW = depthwiseConv2Doptions.StrideW()
        self.strideH = depthwiseConv2Doptions.StrideH()
        self.depthMultiplier = depthwiseConv2Doptions.DepthMultiplier()
        self.fusedActivationFunction = depthwiseConv2Doptions.FusedActivationFunction()
        self.dilationWFactor = depthwiseConv2Doptions.DilationWFactor()
        self.dilationHFactor = depthwiseConv2Doptions.DilationHFactor()

    def Pack(self, builder):
        DepthwiseConv2DOptionsStart(builder)
        DepthwiseConv2DOptionsAddPadding(builder, self.padding)
        DepthwiseConv2DOptionsAddStrideW(builder, self.strideW)
        DepthwiseConv2DOptionsAddStrideH(builder, self.strideH)
        DepthwiseConv2DOptionsAddDepthMultiplier(builder, self.depthMultiplier)
        DepthwiseConv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        DepthwiseConv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        DepthwiseConv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        depthwiseConv2Doptions = DepthwiseConv2DOptionsEnd(builder)
        return depthwiseConv2Doptions