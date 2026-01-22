import flatbuffers
from flatbuffers.compat import import_numpy
class DivOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        divOptions = DivOptions()
        divOptions.Init(buf, pos)
        return cls.InitFromObj(divOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, divOptions):
        x = DivOptionsT()
        x._UnPack(divOptions)
        return x

    def _UnPack(self, divOptions):
        if divOptions is None:
            return
        self.fusedActivationFunction = divOptions.FusedActivationFunction()

    def Pack(self, builder):
        DivOptionsStart(builder)
        DivOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        divOptions = DivOptionsEnd(builder)
        return divOptions