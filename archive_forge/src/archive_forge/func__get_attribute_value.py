from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _get_attribute_value(node, name, default=None, error=None):
    if node.hasAttribute(name):
        return _decode(node.attributes[name].value)
    elif error:
        raise ValueError(error)
    else:
        return default