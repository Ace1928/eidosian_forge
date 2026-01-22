import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
class GenericContent:
    """Abstract base class for AMQP content.

    Subclasses should override the PROPERTIES attribute.
    """
    CLASS_ID = None
    PROPERTIES = [('dummy', 's')]

    def __init__(self, frame_method=None, frame_args=None, **props):
        self.frame_method = frame_method
        self.frame_args = frame_args
        self.properties = props
        self._pending_chunks = []
        self.body_received = 0
        self.body_size = 0
        self.ready = False
    __slots__ = ('frame_method', 'frame_args', 'properties', '_pending_chunks', 'body_received', 'body_size', 'ready', '__dict__', '__weakref__')

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError('__setstate__')
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(name)

    def _load_properties(self, class_id, buf, offset):
        """Load AMQP properties.

        Given the raw bytes containing the property-flags and property-list
        from a content-frame-header, parse and insert into a dictionary
        stored in this object as an attribute named 'properties'.
        """
        props, offset = PROPERTY_CLASSES[class_id](buf, offset)
        self.properties = props
        return offset

    def _serialize_properties(self):
        """Serialize AMQP properties.

        Serialize the 'properties' attribute (a dictionary) into
        the raw bytes making up a set of property flags and a
        property list, suitable for putting into a content frame header.
        """
        shift = 15
        flag_bits = 0
        flags = []
        sformat, svalues = ([], [])
        props = self.properties
        for key, proptype in self.PROPERTIES:
            val = props.get(key, None)
            if val is not None:
                if shift == 0:
                    flags.append(flag_bits)
                    flag_bits = 0
                    shift = 15
                flag_bits |= 1 << shift
                if proptype != 'bit':
                    sformat.append(str_to_bytes(proptype))
                    svalues.append(val)
            shift -= 1
        flags.append(flag_bits)
        result = BytesIO()
        write = result.write
        for flag_bits in flags:
            write(pack('>H', flag_bits))
        write(dumps(b''.join(sformat), svalues))
        return result.getvalue()

    def inbound_header(self, buf, offset=0):
        class_id, self.body_size = unpack_from('>HxxQ', buf, offset)
        offset += 12
        self._load_properties(class_id, buf, offset)
        if not self.body_size:
            self.ready = True
        return offset

    def inbound_body(self, buf):
        chunks = self._pending_chunks
        self.body_received += len(buf)
        if self.body_received >= self.body_size:
            if chunks:
                chunks.append(buf)
                self.body = bytes().join(chunks)
                chunks[:] = []
            else:
                self.body = buf
            self.ready = True
        else:
            chunks.append(buf)