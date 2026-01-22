import flatbuffers
from flatbuffers.compat import import_numpy
class SubOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.potScaleInt16 = True

    @classmethod
    def InitFromBuf(cls, buf, pos):
        subOptions = SubOptions()
        subOptions.Init(buf, pos)
        return cls.InitFromObj(subOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, subOptions):
        x = SubOptionsT()
        x._UnPack(subOptions)
        return x

    def _UnPack(self, subOptions):
        if subOptions is None:
            return
        self.fusedActivationFunction = subOptions.FusedActivationFunction()
        self.potScaleInt16 = subOptions.PotScaleInt16()

    def Pack(self, builder):
        SubOptionsStart(builder)
        SubOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SubOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        subOptions = SubOptionsEnd(builder)
        return subOptions