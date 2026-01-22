from os_ken.lib import stringify
from lxml import objectify
import lxml.etree as ET
class _Unimpl(_Base):
    _ELEMENTS = [_e('raw_et', is_list=False)]

    def to_et(self, tag):
        assert self.raw_et.tag == tag
        return self.raw_et

    @classmethod
    def from_et(cls, et):
        return cls(raw_et=et)