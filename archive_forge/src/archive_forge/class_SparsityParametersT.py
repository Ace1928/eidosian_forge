import flatbuffers
from flatbuffers.compat import import_numpy
class SparsityParametersT(object):

    def __init__(self):
        self.traversalOrder = None
        self.blockMap = None
        self.dimMetadata = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparsityParameters = SparsityParameters()
        sparsityParameters.Init(buf, pos)
        return cls.InitFromObj(sparsityParameters)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, sparsityParameters):
        x = SparsityParametersT()
        x._UnPack(sparsityParameters)
        return x

    def _UnPack(self, sparsityParameters):
        if sparsityParameters is None:
            return
        if not sparsityParameters.TraversalOrderIsNone():
            if np is None:
                self.traversalOrder = []
                for i in range(sparsityParameters.TraversalOrderLength()):
                    self.traversalOrder.append(sparsityParameters.TraversalOrder(i))
            else:
                self.traversalOrder = sparsityParameters.TraversalOrderAsNumpy()
        if not sparsityParameters.BlockMapIsNone():
            if np is None:
                self.blockMap = []
                for i in range(sparsityParameters.BlockMapLength()):
                    self.blockMap.append(sparsityParameters.BlockMap(i))
            else:
                self.blockMap = sparsityParameters.BlockMapAsNumpy()
        if not sparsityParameters.DimMetadataIsNone():
            self.dimMetadata = []
            for i in range(sparsityParameters.DimMetadataLength()):
                if sparsityParameters.DimMetadata(i) is None:
                    self.dimMetadata.append(None)
                else:
                    dimensionMetadata_ = DimensionMetadataT.InitFromObj(sparsityParameters.DimMetadata(i))
                    self.dimMetadata.append(dimensionMetadata_)

    def Pack(self, builder):
        if self.traversalOrder is not None:
            if np is not None and type(self.traversalOrder) is np.ndarray:
                traversalOrder = builder.CreateNumpyVector(self.traversalOrder)
            else:
                SparsityParametersStartTraversalOrderVector(builder, len(self.traversalOrder))
                for i in reversed(range(len(self.traversalOrder))):
                    builder.PrependInt32(self.traversalOrder[i])
                traversalOrder = builder.EndVector()
        if self.blockMap is not None:
            if np is not None and type(self.blockMap) is np.ndarray:
                blockMap = builder.CreateNumpyVector(self.blockMap)
            else:
                SparsityParametersStartBlockMapVector(builder, len(self.blockMap))
                for i in reversed(range(len(self.blockMap))):
                    builder.PrependInt32(self.blockMap[i])
                blockMap = builder.EndVector()
        if self.dimMetadata is not None:
            dimMetadatalist = []
            for i in range(len(self.dimMetadata)):
                dimMetadatalist.append(self.dimMetadata[i].Pack(builder))
            SparsityParametersStartDimMetadataVector(builder, len(self.dimMetadata))
            for i in reversed(range(len(self.dimMetadata))):
                builder.PrependUOffsetTRelative(dimMetadatalist[i])
            dimMetadata = builder.EndVector()
        SparsityParametersStart(builder)
        if self.traversalOrder is not None:
            SparsityParametersAddTraversalOrder(builder, traversalOrder)
        if self.blockMap is not None:
            SparsityParametersAddBlockMap(builder, blockMap)
        if self.dimMetadata is not None:
            SparsityParametersAddDimMetadata(builder, dimMetadata)
        sparsityParameters = SparsityParametersEnd(builder)
        return sparsityParameters