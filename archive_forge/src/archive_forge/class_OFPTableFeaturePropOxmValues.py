import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_PACKET_TYPES)
class OFPTableFeaturePropOxmValues(OFPTableFeatureProp):

    def __init__(self, type_=None, length=None, _ordered_values=None, **kwargs):
        super(OFPTableFeaturePropOxmValues, self).__init__(type_, length)
        if _ordered_values is not None:
            assert not kwargs
            self.oxm_values = _ordered_values
        else:
            kwargs = dict((ofproto.oxm_normalize_user(k, v) for k, v in kwargs.items()))
            values = [ofproto.oxm_from_user(k, v) for k, v in kwargs.items()]
            values.sort(key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0])
            self.oxm_values = [ofproto.oxm_to_user(n, v, m) for n, v, m in values]

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        values = []
        while rest:
            n, value, mask, field_len = ofproto.oxm_parse(rest, 0)
            k, uv = ofproto.oxm_to_user(n, value, mask)
            values.append((k, uv))
            rest = rest[field_len:]
        return cls(_ordered_values=values)

    def serialize_body(self):
        values = [ofproto.oxm_from_user(k, uv) for k, uv in self.oxm_values]
        offset = 0
        buf = bytearray()
        for n, value, mask in values:
            offset += ofproto.oxm_serialize(n, value, mask, buf, offset)
        return buf

    def __getitem__(self, key):
        return dict(self.oxm_values)[key]

    def __contains__(self, key):
        return key in dict(self.oxm_values)

    def iteritems(self):
        return iter(dict(self.oxm_values).items())

    def items(self):
        return self.oxm_values

    def get(self, key, default=None):
        return dict(self.oxm_values).get(key, default)

    def stringify_attrs(self):
        yield ('oxm_values', dict(self.oxm_values))

    def to_jsondict(self):
        """
        Returns a dict expressing the OXM values.
        """
        body = {'oxm_values': [ofproto.oxm_to_jsondict(k, uv) for k, uv in self.oxm_values], 'length': self.length, 'type': self.type}
        return {self.__class__.__name__: body}

    @classmethod
    def from_jsondict(cls, dict_):
        """
        Returns an object which is generated from a dict.
        Exception raises:
        KeyError -- Unknown OXM value is defined in dict
        """
        type_ = dict_['type']
        values = [ofproto.oxm_from_jsondict(f) for f in dict_['oxm_values']]
        return cls(type_=type_, _ordered_values=values)