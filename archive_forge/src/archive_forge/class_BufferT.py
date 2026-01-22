import flatbuffers
from flatbuffers.compat import import_numpy
class BufferT(object):

    def __init__(self):
        self.data = None
        self.offset = 0
        self.size = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        buffer = Buffer()
        buffer.Init(buf, pos)
        return cls.InitFromObj(buffer)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, buffer):
        x = BufferT()
        x._UnPack(buffer)
        return x

    def _UnPack(self, buffer):
        if buffer is None:
            return
        if not buffer.DataIsNone():
            if np is None:
                self.data = []
                for i in range(buffer.DataLength()):
                    self.data.append(buffer.Data(i))
            else:
                self.data = buffer.DataAsNumpy()
        self.offset = buffer.Offset()
        self.size = buffer.Size()

    def Pack(self, builder):
        if self.data is not None:
            if np is not None and type(self.data) is np.ndarray:
                data = builder.CreateNumpyVector(self.data)
            else:
                BufferStartDataVector(builder, len(self.data))
                for i in reversed(range(len(self.data))):
                    builder.PrependUint8(self.data[i])
                data = builder.EndVector()
        BufferStart(builder)
        if self.data is not None:
            BufferAddData(builder, data)
        BufferAddOffset(builder, self.offset)
        BufferAddSize(builder, self.size)
        buffer = BufferEnd(builder)
        return buffer