import flatbuffers
from flatbuffers.compat import import_numpy
class Int32VectorT(object):

    def __init__(self):
        self.values = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        int32Vector = Int32Vector()
        int32Vector.Init(buf, pos)
        return cls.InitFromObj(int32Vector)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, int32Vector):
        x = Int32VectorT()
        x._UnPack(int32Vector)
        return x

    def _UnPack(self, int32Vector):
        if int32Vector is None:
            return
        if not int32Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(int32Vector.ValuesLength()):
                    self.values.append(int32Vector.Values(i))
            else:
                self.values = int32Vector.ValuesAsNumpy()

    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Int32VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependInt32(self.values[i])
                values = builder.EndVector()
        Int32VectorStart(builder)
        if self.values is not None:
            Int32VectorAddValues(builder, values)
        int32Vector = Int32VectorEnd(builder)
        return int32Vector