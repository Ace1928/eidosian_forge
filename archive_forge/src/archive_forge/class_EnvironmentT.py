import flatbuffers
from flatbuffers.compat import import_numpy
class EnvironmentT(object):

    def __init__(self):
        self.tensorflowVersion = None
        self.apiVersion = 0
        self.modelType = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        environment = Environment()
        environment.Init(buf, pos)
        return cls.InitFromObj(environment)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, environment):
        x = EnvironmentT()
        x._UnPack(environment)
        return x

    def _UnPack(self, environment):
        if environment is None:
            return
        self.tensorflowVersion = environment.TensorflowVersion()
        self.apiVersion = environment.ApiVersion()
        self.modelType = environment.ModelType()

    def Pack(self, builder):
        if self.tensorflowVersion is not None:
            tensorflowVersion = builder.CreateString(self.tensorflowVersion)
        EnvironmentStart(builder)
        if self.tensorflowVersion is not None:
            EnvironmentAddTensorflowVersion(builder, tensorflowVersion)
        EnvironmentAddApiVersion(builder, self.apiVersion)
        EnvironmentAddModelType(builder, self.modelType)
        environment = EnvironmentEnd(builder)
        return environment