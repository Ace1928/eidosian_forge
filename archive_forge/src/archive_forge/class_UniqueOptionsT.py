import flatbuffers
from flatbuffers.compat import import_numpy
class UniqueOptionsT(object):

    def __init__(self):
        self.idxOutType = 2

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uniqueOptions = UniqueOptions()
        uniqueOptions.Init(buf, pos)
        return cls.InitFromObj(uniqueOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, uniqueOptions):
        x = UniqueOptionsT()
        x._UnPack(uniqueOptions)
        return x

    def _UnPack(self, uniqueOptions):
        if uniqueOptions is None:
            return
        self.idxOutType = uniqueOptions.IdxOutType()

    def Pack(self, builder):
        UniqueOptionsStart(builder)
        UniqueOptionsAddIdxOutType(builder, self.idxOutType)
        uniqueOptions = UniqueOptionsEnd(builder)
        return uniqueOptions