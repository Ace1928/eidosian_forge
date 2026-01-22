from thrift.Thrift import TType, TMessageType, TException
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TProtocol
class BinaryAnnotation:
    """
  Attributes:
   - key
   - value
   - annotation_type
   - host
  """
    thrift_spec = (None, (1, TType.STRING, 'key', None, None), (2, TType.STRING, 'value', None, None), (3, TType.I32, 'annotation_type', None, None), (4, TType.STRUCT, 'host', (Endpoint, Endpoint.thrift_spec), None))

    def __init__(self, key=None, value=None, annotation_type=None, host=None):
        self.key = key
        self.value = value
        self.annotation_type = annotation_type
        self.host = host

    def read(self, iprot):
        if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and (self.thrift_spec is not None) and (fastbinary is not None):
            fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
            return
        iprot.readStructBegin()
        while True:
            fname, ftype, fid = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.key = iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.value = iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.I32:
                    self.annotation_type = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRUCT:
                    self.host = Endpoint()
                    self.host.read(iprot)
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and (fastbinary is not None):
            oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
            return
        oprot.writeStructBegin('BinaryAnnotation')
        if self.key is not None:
            oprot.writeFieldBegin('key', TType.STRING, 1)
            oprot.writeString(self.key)
            oprot.writeFieldEnd()
        if self.value is not None:
            oprot.writeFieldBegin('value', TType.STRING, 2)
            oprot.writeString(self.value)
            oprot.writeFieldEnd()
        if self.annotation_type is not None:
            oprot.writeFieldBegin('annotation_type', TType.I32, 3)
            oprot.writeI32(self.annotation_type)
            oprot.writeFieldEnd()
        if self.host is not None:
            oprot.writeFieldBegin('host', TType.STRUCT, 4)
            self.host.write(oprot)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value) for key, value in self.__dict__.iteritems()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other