import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
class TomlArraySeparatorEncoder(TomlEncoder):

    def __init__(self, _dict=dict, preserve=False, separator=','):
        super(TomlArraySeparatorEncoder, self).__init__(_dict, preserve)
        if separator.strip() == '':
            separator = ',' + separator
        elif separator.strip(' \t\n\r,'):
            raise ValueError('Invalid separator for arrays')
        self.separator = separator

    def dump_list(self, v):
        t = []
        retval = '['
        for u in v:
            t.append(self.dump_value(u))
        while t != []:
            s = []
            for u in t:
                if isinstance(u, list):
                    for r in u:
                        s.append(r)
                else:
                    retval += ' ' + unicode(u) + self.separator
            t = s
        retval += ']'
        return retval