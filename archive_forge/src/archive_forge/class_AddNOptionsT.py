import flatbuffers
from flatbuffers.compat import import_numpy
class AddNOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        addNoptions = AddNOptions()
        addNoptions.Init(buf, pos)
        return cls.InitFromObj(addNoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, addNoptions):
        x = AddNOptionsT()
        x._UnPack(addNoptions)
        return x

    def _UnPack(self, addNoptions):
        if addNoptions is None:
            return

    def Pack(self, builder):
        AddNOptionsStart(builder)
        addNoptions = AddNOptionsEnd(builder)
        return addNoptions