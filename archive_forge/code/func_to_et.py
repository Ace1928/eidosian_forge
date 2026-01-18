from os_ken.lib import stringify
from lxml import objectify
import lxml.etree as ET
def to_et(self, tag):
    assert self.raw_et.tag == tag
    return self.raw_et