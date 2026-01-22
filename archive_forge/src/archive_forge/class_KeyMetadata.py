import functools
import re
from ovs.flow.decoders import decode_default
class KeyMetadata(object):
    """Class for keeping key metadata.

    Attributes:
        kpos (int): The position of the keyword in the parent string.
        vpos (int): The position of the value in the parent string.
        kstring (string): The keyword string as found in the flow string.
        vstring (string): The value as found in the flow string.
        delim (string): Optional, the string used as delimiter between the key
            and the value.
        end_delim (string): Optional, the string used as end delimiter
    """

    def __init__(self, kpos, vpos, kstring, vstring, delim='', end_delim=''):
        """Constructor."""
        self.kpos = kpos
        self.vpos = vpos
        self.kstring = kstring
        self.vstring = vstring
        self.delim = delim
        self.end_delim = end_delim

    def __str__(self):
        return 'key: [{},{}), val:[{}, {})'.format(self.kpos, self.kpos + len(self.kstring), self.vpos, self.vpos + len(self.vstring))

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self)