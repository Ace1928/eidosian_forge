import flatbuffers
from flatbuffers.compat import import_numpy
class WhereOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        whereOptions = WhereOptions()
        whereOptions.Init(buf, pos)
        return cls.InitFromObj(whereOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, whereOptions):
        x = WhereOptionsT()
        x._UnPack(whereOptions)
        return x

    def _UnPack(self, whereOptions):
        if whereOptions is None:
            return

    def Pack(self, builder):
        WhereOptionsStart(builder)
        whereOptions = WhereOptionsEnd(builder)
        return whereOptions