import lxml.etree as ET
from functools import partial
def add_dict(elem, item):
    attrib = elem.attrib
    for k, v in item.items():
        if isinstance(v, basestring):
            attrib[k] = v
        else:
            attrib[k] = typemap[type(v)](None, v)