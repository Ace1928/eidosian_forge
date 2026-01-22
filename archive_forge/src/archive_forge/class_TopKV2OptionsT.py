import flatbuffers
from flatbuffers.compat import import_numpy
class TopKV2OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        topKv2Options = TopKV2Options()
        topKv2Options.Init(buf, pos)
        return cls.InitFromObj(topKv2Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, topKv2Options):
        x = TopKV2OptionsT()
        x._UnPack(topKv2Options)
        return x

    def _UnPack(self, topKv2Options):
        if topKv2Options is None:
            return

    def Pack(self, builder):
        TopKV2OptionsStart(builder)
        topKv2Options = TopKV2OptionsEnd(builder)
        return topKv2Options