import flatbuffers
from flatbuffers.compat import import_numpy
class SkipGramOptionsT(object):

    def __init__(self):
        self.ngramSize = 0
        self.maxSkipSize = 0
        self.includeAllNgrams = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        skipGramOptions = SkipGramOptions()
        skipGramOptions.Init(buf, pos)
        return cls.InitFromObj(skipGramOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, skipGramOptions):
        x = SkipGramOptionsT()
        x._UnPack(skipGramOptions)
        return x

    def _UnPack(self, skipGramOptions):
        if skipGramOptions is None:
            return
        self.ngramSize = skipGramOptions.NgramSize()
        self.maxSkipSize = skipGramOptions.MaxSkipSize()
        self.includeAllNgrams = skipGramOptions.IncludeAllNgrams()

    def Pack(self, builder):
        SkipGramOptionsStart(builder)
        SkipGramOptionsAddNgramSize(builder, self.ngramSize)
        SkipGramOptionsAddMaxSkipSize(builder, self.maxSkipSize)
        SkipGramOptionsAddIncludeAllNgrams(builder, self.includeAllNgrams)
        skipGramOptions = SkipGramOptionsEnd(builder)
        return skipGramOptions