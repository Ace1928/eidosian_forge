import flatbuffers
from flatbuffers.compat import import_numpy
class MulOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        mulOptions = MulOptions()
        mulOptions.Init(buf, pos)
        return cls.InitFromObj(mulOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, mulOptions):
        x = MulOptionsT()
        x._UnPack(mulOptions)
        return x

    def _UnPack(self, mulOptions):
        if mulOptions is None:
            return
        self.fusedActivationFunction = mulOptions.FusedActivationFunction()

    def Pack(self, builder):
        MulOptionsStart(builder)
        MulOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        mulOptions = MulOptionsEnd(builder)
        return mulOptions