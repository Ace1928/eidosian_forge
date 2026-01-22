import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class OFPTableFeaturesStats(StringifyMixin):
    _TYPE = {'utf-8': ['name']}

    def __init__(self, table_id=None, name=None, metadata_match=None, metadata_write=None, config=None, max_entries=None, properties=None, length=None):
        super(OFPTableFeaturesStats, self).__init__()
        self.length = None
        self.table_id = table_id
        self.name = name
        self.metadata_match = metadata_match
        self.metadata_write = metadata_write
        self.config = config
        self.max_entries = max_entries
        self.properties = properties

    @classmethod
    def parser(cls, buf, offset):
        table_features = cls()
        table_features.length, table_features.table_id, name, table_features.metadata_match, table_features.metadata_write, table_features.config, table_features.max_entries = struct.unpack_from(ofproto.OFP_TABLE_FEATURES_PACK_STR, buf, offset)
        table_features.name = name.rstrip(b'\x00')
        props = []
        rest = buf[offset + ofproto.OFP_TABLE_FEATURES_SIZE:offset + table_features.length]
        while rest:
            p, rest = OFPTableFeatureProp.parse(rest)
            props.append(p)
        table_features.properties = props
        return table_features

    def serialize(self):
        bin_props = bytearray()
        for p in self.properties:
            bin_props += p.serialize()
        self.length = ofproto.OFP_TABLE_FEATURES_SIZE + len(bin_props)
        buf = bytearray()
        msg_pack_into(ofproto.OFP_TABLE_FEATURES_PACK_STR, buf, 0, self.length, self.table_id, self.name, self.metadata_match, self.metadata_write, self.config, self.max_entries)
        return buf + bin_props