import flatbuffers
from flatbuffers.compat import import_numpy
class SpaceToBatchNDOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        spaceToBatchNdoptions = SpaceToBatchNDOptions()
        spaceToBatchNdoptions.Init(buf, pos)
        return cls.InitFromObj(spaceToBatchNdoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, spaceToBatchNdoptions):
        x = SpaceToBatchNDOptionsT()
        x._UnPack(spaceToBatchNdoptions)
        return x

    def _UnPack(self, spaceToBatchNdoptions):
        if spaceToBatchNdoptions is None:
            return

    def Pack(self, builder):
        SpaceToBatchNDOptionsStart(builder)
        spaceToBatchNdoptions = SpaceToBatchNDOptionsEnd(builder)
        return spaceToBatchNdoptions