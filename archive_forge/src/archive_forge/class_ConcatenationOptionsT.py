import flatbuffers
from flatbuffers.compat import import_numpy
class ConcatenationOptionsT(object):

    def __init__(self):
        self.axis = 0
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        concatenationOptions = ConcatenationOptions()
        concatenationOptions.Init(buf, pos)
        return cls.InitFromObj(concatenationOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, concatenationOptions):
        x = ConcatenationOptionsT()
        x._UnPack(concatenationOptions)
        return x

    def _UnPack(self, concatenationOptions):
        if concatenationOptions is None:
            return
        self.axis = concatenationOptions.Axis()
        self.fusedActivationFunction = concatenationOptions.FusedActivationFunction()

    def Pack(self, builder):
        ConcatenationOptionsStart(builder)
        ConcatenationOptionsAddAxis(builder, self.axis)
        ConcatenationOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        concatenationOptions = ConcatenationOptionsEnd(builder)
        return concatenationOptions