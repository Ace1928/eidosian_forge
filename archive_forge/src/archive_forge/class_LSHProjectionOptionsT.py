import flatbuffers
from flatbuffers.compat import import_numpy
class LSHProjectionOptionsT(object):

    def __init__(self):
        self.type = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lshprojectionOptions = LSHProjectionOptions()
        lshprojectionOptions.Init(buf, pos)
        return cls.InitFromObj(lshprojectionOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, lshprojectionOptions):
        x = LSHProjectionOptionsT()
        x._UnPack(lshprojectionOptions)
        return x

    def _UnPack(self, lshprojectionOptions):
        if lshprojectionOptions is None:
            return
        self.type = lshprojectionOptions.Type()

    def Pack(self, builder):
        LSHProjectionOptionsStart(builder)
        LSHProjectionOptionsAddType(builder, self.type)
        lshprojectionOptions = LSHProjectionOptionsEnd(builder)
        return lshprojectionOptions