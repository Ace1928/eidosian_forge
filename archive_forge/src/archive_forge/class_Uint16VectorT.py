import flatbuffers
from flatbuffers.compat import import_numpy
class Uint16VectorT(object):

    def __init__(self):
        self.values = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uint16Vector = Uint16Vector()
        uint16Vector.Init(buf, pos)
        return cls.InitFromObj(uint16Vector)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, uint16Vector):
        x = Uint16VectorT()
        x._UnPack(uint16Vector)
        return x

    def _UnPack(self, uint16Vector):
        if uint16Vector is None:
            return
        if not uint16Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(uint16Vector.ValuesLength()):
                    self.values.append(uint16Vector.Values(i))
            else:
                self.values = uint16Vector.ValuesAsNumpy()

    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Uint16VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint16(self.values[i])
                values = builder.EndVector()
        Uint16VectorStart(builder)
        if self.values is not None:
            Uint16VectorAddValues(builder, values)
        uint16Vector = Uint16VectorEnd(builder)
        return uint16Vector