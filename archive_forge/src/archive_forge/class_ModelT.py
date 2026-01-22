import flatbuffers
from flatbuffers.compat import import_numpy
class ModelT(object):

    def __init__(self):
        self.version = 0
        self.operatorCodes = None
        self.subgraphs = None
        self.description = None
        self.buffers = None
        self.metadataBuffer = None
        self.metadata = None
        self.signatureDefs = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        model = Model()
        model.Init(buf, pos)
        return cls.InitFromObj(model)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, model):
        x = ModelT()
        x._UnPack(model)
        return x

    def _UnPack(self, model):
        if model is None:
            return
        self.version = model.Version()
        if not model.OperatorCodesIsNone():
            self.operatorCodes = []
            for i in range(model.OperatorCodesLength()):
                if model.OperatorCodes(i) is None:
                    self.operatorCodes.append(None)
                else:
                    operatorCode_ = OperatorCodeT.InitFromObj(model.OperatorCodes(i))
                    self.operatorCodes.append(operatorCode_)
        if not model.SubgraphsIsNone():
            self.subgraphs = []
            for i in range(model.SubgraphsLength()):
                if model.Subgraphs(i) is None:
                    self.subgraphs.append(None)
                else:
                    subGraph_ = SubGraphT.InitFromObj(model.Subgraphs(i))
                    self.subgraphs.append(subGraph_)
        self.description = model.Description()
        if not model.BuffersIsNone():
            self.buffers = []
            for i in range(model.BuffersLength()):
                if model.Buffers(i) is None:
                    self.buffers.append(None)
                else:
                    buffer_ = BufferT.InitFromObj(model.Buffers(i))
                    self.buffers.append(buffer_)
        if not model.MetadataBufferIsNone():
            if np is None:
                self.metadataBuffer = []
                for i in range(model.MetadataBufferLength()):
                    self.metadataBuffer.append(model.MetadataBuffer(i))
            else:
                self.metadataBuffer = model.MetadataBufferAsNumpy()
        if not model.MetadataIsNone():
            self.metadata = []
            for i in range(model.MetadataLength()):
                if model.Metadata(i) is None:
                    self.metadata.append(None)
                else:
                    metadata_ = MetadataT.InitFromObj(model.Metadata(i))
                    self.metadata.append(metadata_)
        if not model.SignatureDefsIsNone():
            self.signatureDefs = []
            for i in range(model.SignatureDefsLength()):
                if model.SignatureDefs(i) is None:
                    self.signatureDefs.append(None)
                else:
                    signatureDef_ = SignatureDefT.InitFromObj(model.SignatureDefs(i))
                    self.signatureDefs.append(signatureDef_)

    def Pack(self, builder):
        if self.operatorCodes is not None:
            operatorCodeslist = []
            for i in range(len(self.operatorCodes)):
                operatorCodeslist.append(self.operatorCodes[i].Pack(builder))
            ModelStartOperatorCodesVector(builder, len(self.operatorCodes))
            for i in reversed(range(len(self.operatorCodes))):
                builder.PrependUOffsetTRelative(operatorCodeslist[i])
            operatorCodes = builder.EndVector()
        if self.subgraphs is not None:
            subgraphslist = []
            for i in range(len(self.subgraphs)):
                subgraphslist.append(self.subgraphs[i].Pack(builder))
            ModelStartSubgraphsVector(builder, len(self.subgraphs))
            for i in reversed(range(len(self.subgraphs))):
                builder.PrependUOffsetTRelative(subgraphslist[i])
            subgraphs = builder.EndVector()
        if self.description is not None:
            description = builder.CreateString(self.description)
        if self.buffers is not None:
            bufferslist = []
            for i in range(len(self.buffers)):
                bufferslist.append(self.buffers[i].Pack(builder))
            ModelStartBuffersVector(builder, len(self.buffers))
            for i in reversed(range(len(self.buffers))):
                builder.PrependUOffsetTRelative(bufferslist[i])
            buffers = builder.EndVector()
        if self.metadataBuffer is not None:
            if np is not None and type(self.metadataBuffer) is np.ndarray:
                metadataBuffer = builder.CreateNumpyVector(self.metadataBuffer)
            else:
                ModelStartMetadataBufferVector(builder, len(self.metadataBuffer))
                for i in reversed(range(len(self.metadataBuffer))):
                    builder.PrependInt32(self.metadataBuffer[i])
                metadataBuffer = builder.EndVector()
        if self.metadata is not None:
            metadatalist = []
            for i in range(len(self.metadata)):
                metadatalist.append(self.metadata[i].Pack(builder))
            ModelStartMetadataVector(builder, len(self.metadata))
            for i in reversed(range(len(self.metadata))):
                builder.PrependUOffsetTRelative(metadatalist[i])
            metadata = builder.EndVector()
        if self.signatureDefs is not None:
            signatureDefslist = []
            for i in range(len(self.signatureDefs)):
                signatureDefslist.append(self.signatureDefs[i].Pack(builder))
            ModelStartSignatureDefsVector(builder, len(self.signatureDefs))
            for i in reversed(range(len(self.signatureDefs))):
                builder.PrependUOffsetTRelative(signatureDefslist[i])
            signatureDefs = builder.EndVector()
        ModelStart(builder)
        ModelAddVersion(builder, self.version)
        if self.operatorCodes is not None:
            ModelAddOperatorCodes(builder, operatorCodes)
        if self.subgraphs is not None:
            ModelAddSubgraphs(builder, subgraphs)
        if self.description is not None:
            ModelAddDescription(builder, description)
        if self.buffers is not None:
            ModelAddBuffers(builder, buffers)
        if self.metadataBuffer is not None:
            ModelAddMetadataBuffer(builder, metadataBuffer)
        if self.metadata is not None:
            ModelAddMetadata(builder, metadata)
        if self.signatureDefs is not None:
            ModelAddSignatureDefs(builder, signatureDefs)
        model = ModelEnd(builder)
        return model