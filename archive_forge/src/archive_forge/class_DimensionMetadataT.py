import flatbuffers
from flatbuffers.compat import import_numpy
class DimensionMetadataT(object):

    def __init__(self):
        self.format = 0
        self.denseSize = 0
        self.arraySegmentsType = 0
        self.arraySegments = None
        self.arrayIndicesType = 0
        self.arrayIndices = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dimensionMetadata = DimensionMetadata()
        dimensionMetadata.Init(buf, pos)
        return cls.InitFromObj(dimensionMetadata)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, dimensionMetadata):
        x = DimensionMetadataT()
        x._UnPack(dimensionMetadata)
        return x

    def _UnPack(self, dimensionMetadata):
        if dimensionMetadata is None:
            return
        self.format = dimensionMetadata.Format()
        self.denseSize = dimensionMetadata.DenseSize()
        self.arraySegmentsType = dimensionMetadata.ArraySegmentsType()
        self.arraySegments = SparseIndexVectorCreator(self.arraySegmentsType, dimensionMetadata.ArraySegments())
        self.arrayIndicesType = dimensionMetadata.ArrayIndicesType()
        self.arrayIndices = SparseIndexVectorCreator(self.arrayIndicesType, dimensionMetadata.ArrayIndices())

    def Pack(self, builder):
        if self.arraySegments is not None:
            arraySegments = self.arraySegments.Pack(builder)
        if self.arrayIndices is not None:
            arrayIndices = self.arrayIndices.Pack(builder)
        DimensionMetadataStart(builder)
        DimensionMetadataAddFormat(builder, self.format)
        DimensionMetadataAddDenseSize(builder, self.denseSize)
        DimensionMetadataAddArraySegmentsType(builder, self.arraySegmentsType)
        if self.arraySegments is not None:
            DimensionMetadataAddArraySegments(builder, arraySegments)
        DimensionMetadataAddArrayIndicesType(builder, self.arrayIndicesType)
        if self.arrayIndices is not None:
            DimensionMetadataAddArrayIndices(builder, arrayIndices)
        dimensionMetadata = DimensionMetadataEnd(builder)
        return dimensionMetadata