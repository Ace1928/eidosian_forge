import flatbuffers
from flatbuffers.compat import import_numpy
class DynamicUpdateSliceOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dynamicUpdateSliceOptions = DynamicUpdateSliceOptions()
        dynamicUpdateSliceOptions.Init(buf, pos)
        return cls.InitFromObj(dynamicUpdateSliceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, dynamicUpdateSliceOptions):
        x = DynamicUpdateSliceOptionsT()
        x._UnPack(dynamicUpdateSliceOptions)
        return x

    def _UnPack(self, dynamicUpdateSliceOptions):
        if dynamicUpdateSliceOptions is None:
            return

    def Pack(self, builder):
        DynamicUpdateSliceOptionsStart(builder)
        dynamicUpdateSliceOptions = DynamicUpdateSliceOptionsEnd(builder)
        return dynamicUpdateSliceOptions