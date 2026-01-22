import flatbuffers
from flatbuffers.compat import import_numpy
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