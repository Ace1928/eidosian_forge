import flatbuffers
from flatbuffers.compat import import_numpy
def BuiltinOptionsCreator(unionType, table):
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == BuiltinOptions().Conv2DOptions:
        return Conv2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DepthwiseConv2DOptions:
        return DepthwiseConv2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ConcatEmbeddingsOptions:
        return ConcatEmbeddingsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LSHProjectionOptions:
        return LSHProjectionOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().Pool2DOptions:
        return Pool2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SVDFOptions:
        return SVDFOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RNNOptions:
        return RNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FullyConnectedOptions:
        return FullyConnectedOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SoftmaxOptions:
        return SoftmaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ConcatenationOptions:
        return ConcatenationOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AddOptions:
        return AddOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().L2NormOptions:
        return L2NormOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LocalResponseNormalizationOptions:
        return LocalResponseNormalizationOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LSTMOptions:
        return LSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ResizeBilinearOptions:
        return ResizeBilinearOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CallOptions:
        return CallOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReshapeOptions:
        return ReshapeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SkipGramOptions:
        return SkipGramOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SpaceToDepthOptions:
        return SpaceToDepthOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().EmbeddingLookupSparseOptions:
        return EmbeddingLookupSparseOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MulOptions:
        return MulOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PadOptions:
        return PadOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GatherOptions:
        return GatherOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BatchToSpaceNDOptions:
        return BatchToSpaceNDOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SpaceToBatchNDOptions:
        return SpaceToBatchNDOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TransposeOptions:
        return TransposeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReducerOptions:
        return ReducerOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SubOptions:
        return SubOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DivOptions:
        return DivOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SqueezeOptions:
        return SqueezeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SequenceRNNOptions:
        return SequenceRNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().StridedSliceOptions:
        return StridedSliceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ExpOptions:
        return ExpOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TopKV2Options:
        return TopKV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SplitOptions:
        return SplitOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogSoftmaxOptions:
        return LogSoftmaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CastOptions:
        return CastOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DequantizeOptions:
        return DequantizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MaximumMinimumOptions:
        return MaximumMinimumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ArgMaxOptions:
        return ArgMaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LessOptions:
        return LessOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NegOptions:
        return NegOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PadV2Options:
        return PadV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GreaterOptions:
        return GreaterOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GreaterEqualOptions:
        return GreaterEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LessEqualOptions:
        return LessEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SelectOptions:
        return SelectOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SliceOptions:
        return SliceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TransposeConvOptions:
        return TransposeConvOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SparseToDenseOptions:
        return SparseToDenseOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TileOptions:
        return TileOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ExpandDimsOptions:
        return ExpandDimsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().EqualOptions:
        return EqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NotEqualOptions:
        return NotEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ShapeOptions:
        return ShapeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PowOptions:
        return PowOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ArgMinOptions:
        return ArgMinOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FakeQuantOptions:
        return FakeQuantOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PackOptions:
        return PackOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalOrOptions:
        return LogicalOrOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().OneHotOptions:
        return OneHotOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalAndOptions:
        return LogicalAndOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalNotOptions:
        return LogicalNotOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnpackOptions:
        return UnpackOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FloorDivOptions:
        return FloorDivOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SquareOptions:
        return SquareOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ZerosLikeOptions:
        return ZerosLikeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FillOptions:
        return FillOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BidirectionalSequenceLSTMOptions:
        return BidirectionalSequenceLSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BidirectionalSequenceRNNOptions:
        return BidirectionalSequenceRNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnidirectionalSequenceLSTMOptions:
        return UnidirectionalSequenceLSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FloorModOptions:
        return FloorModOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RangeOptions:
        return RangeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ResizeNearestNeighborOptions:
        return ResizeNearestNeighborOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LeakyReluOptions:
        return LeakyReluOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SquaredDifferenceOptions:
        return SquaredDifferenceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MirrorPadOptions:
        return MirrorPadOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AbsOptions:
        return AbsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SplitVOptions:
        return SplitVOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UniqueOptions:
        return UniqueOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReverseV2Options:
        return ReverseV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AddNOptions:
        return AddNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GatherNdOptions:
        return GatherNdOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CosOptions:
        return CosOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().WhereOptions:
        return WhereOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RankOptions:
        return RankOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReverseSequenceOptions:
        return ReverseSequenceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MatrixDiagOptions:
        return MatrixDiagOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().QuantizeOptions:
        return QuantizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MatrixSetDiagOptions:
        return MatrixSetDiagOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HardSwishOptions:
        return HardSwishOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().IfOptions:
        return IfOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().WhileOptions:
        return WhileOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DepthToSpaceOptions:
        return DepthToSpaceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NonMaxSuppressionV4Options:
        return NonMaxSuppressionV4OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NonMaxSuppressionV5Options:
        return NonMaxSuppressionV5OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ScatterNdOptions:
        return ScatterNdOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SelectV2Options:
        return SelectV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DensifyOptions:
        return DensifyOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SegmentSumOptions:
        return SegmentSumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BatchMatMulOptions:
        return BatchMatMulOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CumsumOptions:
        return CumsumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CallOnceOptions:
        return CallOnceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BroadcastToOptions:
        return BroadcastToOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().Rfft2dOptions:
        return Rfft2dOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().Conv3DOptions:
        return Conv3DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HashtableOptions:
        return HashtableOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HashtableFindOptions:
        return HashtableFindOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HashtableImportOptions:
        return HashtableImportOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HashtableSizeOptions:
        return HashtableSizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().VarHandleOptions:
        return VarHandleOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReadVariableOptions:
        return ReadVariableOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AssignVariableOptions:
        return AssignVariableOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RandomOptions:
        return RandomOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BucketizeOptions:
        return BucketizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GeluOptions:
        return GeluOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DynamicUpdateSliceOptions:
        return DynamicUpdateSliceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnsortedSegmentProdOptions:
        return UnsortedSegmentProdOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnsortedSegmentMaxOptions:
        return UnsortedSegmentMaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnsortedSegmentMinOptions:
        return UnsortedSegmentMinOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnsortedSegmentSumOptions:
        return UnsortedSegmentSumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ATan2Options:
        return ATan2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SignOptions:
        return SignOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BitcastOptions:
        return BitcastOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BitwiseXorOptions:
        return BitwiseXorOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RightShiftOptions:
        return RightShiftOptionsT.InitFromBuf(table.Bytes, table.Pos)
    return None