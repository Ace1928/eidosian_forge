import flatbuffers
from flatbuffers.compat import import_numpy
class CallOnceOptionsT(object):

    def __init__(self):
        self.initSubgraphIndex = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        callOnceOptions = CallOnceOptions()
        callOnceOptions.Init(buf, pos)
        return cls.InitFromObj(callOnceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, callOnceOptions):
        x = CallOnceOptionsT()
        x._UnPack(callOnceOptions)
        return x

    def _UnPack(self, callOnceOptions):
        if callOnceOptions is None:
            return
        self.initSubgraphIndex = callOnceOptions.InitSubgraphIndex()

    def Pack(self, builder):
        CallOnceOptionsStart(builder)
        CallOnceOptionsAddInitSubgraphIndex(builder, self.initSubgraphIndex)
        callOnceOptions = CallOnceOptionsEnd(builder)
        return callOnceOptions