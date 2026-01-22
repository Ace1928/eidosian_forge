from os_ken.ofproto.oxx_fields import (
from os_ken.ofproto import ofproto_common
class _OxmClass(object):

    def __init__(self, name, num, type_):
        self.name = name
        self.oxm_field = num
        self.oxm_type = num | self._class << 7
        self.num = self.oxm_type
        self.type = type_