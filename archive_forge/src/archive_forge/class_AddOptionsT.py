import flatbuffers
from flatbuffers.compat import import_numpy
class AddOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.potScaleInt16 = True

    @classmethod
    def InitFromBuf(cls, buf, pos):
        addOptions = AddOptions()
        addOptions.Init(buf, pos)
        return cls.InitFromObj(addOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, addOptions):
        x = AddOptionsT()
        x._UnPack(addOptions)
        return x

    def _UnPack(self, addOptions):
        if addOptions is None:
            return
        self.fusedActivationFunction = addOptions.FusedActivationFunction()
        self.potScaleInt16 = addOptions.PotScaleInt16()

    def Pack(self, builder):
        AddOptionsStart(builder)
        AddOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        AddOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        addOptions = AddOptionsEnd(builder)
        return addOptions