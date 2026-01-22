from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_utils
from os_ken.ofproto import oxm_fields
from os_ken.ofproto import oxs_fields
from struct import calcsize
def _oxm_tlv_header(class_, field, hasmask, length):
    return class_ << 16 | field << 9 | hasmask << 8 | length