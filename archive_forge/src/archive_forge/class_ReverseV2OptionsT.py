import flatbuffers
from flatbuffers.compat import import_numpy
class ReverseV2OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reverseV2Options = ReverseV2Options()
        reverseV2Options.Init(buf, pos)
        return cls.InitFromObj(reverseV2Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, reverseV2Options):
        x = ReverseV2OptionsT()
        x._UnPack(reverseV2Options)
        return x

    def _UnPack(self, reverseV2Options):
        if reverseV2Options is None:
            return

    def Pack(self, builder):
        ReverseV2OptionsStart(builder)
        reverseV2Options = ReverseV2OptionsEnd(builder)
        return reverseV2Options