import flatbuffers
from flatbuffers.compat import import_numpy
class PadV2OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        padV2Options = PadV2Options()
        padV2Options.Init(buf, pos)
        return cls.InitFromObj(padV2Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, padV2Options):
        x = PadV2OptionsT()
        x._UnPack(padV2Options)
        return x

    def _UnPack(self, padV2Options):
        if padV2Options is None:
            return

    def Pack(self, builder):
        PadV2OptionsStart(builder)
        padV2Options = PadV2OptionsEnd(builder)
        return padV2Options