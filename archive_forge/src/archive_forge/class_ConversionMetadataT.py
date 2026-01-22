import flatbuffers
from flatbuffers.compat import import_numpy
class ConversionMetadataT(object):

    def __init__(self):
        self.environment = None
        self.options = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conversionMetadata = ConversionMetadata()
        conversionMetadata.Init(buf, pos)
        return cls.InitFromObj(conversionMetadata)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, conversionMetadata):
        x = ConversionMetadataT()
        x._UnPack(conversionMetadata)
        return x

    def _UnPack(self, conversionMetadata):
        if conversionMetadata is None:
            return
        if conversionMetadata.Environment() is not None:
            self.environment = EnvironmentT.InitFromObj(conversionMetadata.Environment())
        if conversionMetadata.Options() is not None:
            self.options = ConversionOptionsT.InitFromObj(conversionMetadata.Options())

    def Pack(self, builder):
        if self.environment is not None:
            environment = self.environment.Pack(builder)
        if self.options is not None:
            options = self.options.Pack(builder)
        ConversionMetadataStart(builder)
        if self.environment is not None:
            ConversionMetadataAddEnvironment(builder, environment)
        if self.options is not None:
            ConversionMetadataAddOptions(builder, options)
        conversionMetadata = ConversionMetadataEnd(builder)
        return conversionMetadata