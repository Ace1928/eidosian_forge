import flatbuffers
from flatbuffers.compat import import_numpy
class EmbeddingLookupSparseOptionsT(object):

    def __init__(self):
        self.combiner = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        embeddingLookupSparseOptions = EmbeddingLookupSparseOptions()
        embeddingLookupSparseOptions.Init(buf, pos)
        return cls.InitFromObj(embeddingLookupSparseOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, embeddingLookupSparseOptions):
        x = EmbeddingLookupSparseOptionsT()
        x._UnPack(embeddingLookupSparseOptions)
        return x

    def _UnPack(self, embeddingLookupSparseOptions):
        if embeddingLookupSparseOptions is None:
            return
        self.combiner = embeddingLookupSparseOptions.Combiner()

    def Pack(self, builder):
        EmbeddingLookupSparseOptionsStart(builder)
        EmbeddingLookupSparseOptionsAddCombiner(builder, self.combiner)
        embeddingLookupSparseOptions = EmbeddingLookupSparseOptionsEnd(builder)
        return embeddingLookupSparseOptions