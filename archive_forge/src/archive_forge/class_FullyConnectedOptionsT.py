import flatbuffers
from flatbuffers.compat import import_numpy
class FullyConnectedOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.weightsFormat = 0
        self.keepNumDims = False
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fullyConnectedOptions = FullyConnectedOptions()
        fullyConnectedOptions.Init(buf, pos)
        return cls.InitFromObj(fullyConnectedOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, fullyConnectedOptions):
        x = FullyConnectedOptionsT()
        x._UnPack(fullyConnectedOptions)
        return x

    def _UnPack(self, fullyConnectedOptions):
        if fullyConnectedOptions is None:
            return
        self.fusedActivationFunction = fullyConnectedOptions.FusedActivationFunction()
        self.weightsFormat = fullyConnectedOptions.WeightsFormat()
        self.keepNumDims = fullyConnectedOptions.KeepNumDims()
        self.asymmetricQuantizeInputs = fullyConnectedOptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        FullyConnectedOptionsStart(builder)
        FullyConnectedOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        FullyConnectedOptionsAddWeightsFormat(builder, self.weightsFormat)
        FullyConnectedOptionsAddKeepNumDims(builder, self.keepNumDims)
        FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        fullyConnectedOptions = FullyConnectedOptionsEnd(builder)
        return fullyConnectedOptions