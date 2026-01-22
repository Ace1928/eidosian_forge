import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class MagicRule:
    also = None

    def __init__(self, start, value, mask, word, range):
        self.start = start
        self.value = value
        self.mask = mask
        self.word = word
        self.range = range
    rule_ending_re = re.compile(b'(?:~(\\d+))?(?:\\+(\\d+))?\\n$')

    @classmethod
    def from_file(cls, f):
        """Read a rule from the binary magics file. Returns a 2-tuple of
        the nesting depth and the MagicRule."""
        line = f.readline()
        nest_depth, line = line.split(b'>', 1)
        nest_depth = int(nest_depth) if nest_depth else 0
        start, line = line.split(b'=', 1)
        start = int(start)
        if line == b'__NOMAGIC__\n':
            raise DiscardMagicRules
        if sys.version_info[0] >= 3:
            lenvalue = int.from_bytes(line[:2], byteorder='big')
        else:
            lenvalue = (ord(line[0]) << 8) + ord(line[1])
        line = line[2:]
        while len(line) <= lenvalue:
            line += f.readline()
        value, line = (line[:lenvalue], line[lenvalue:])
        if line.startswith(b'&'):
            while len(line) <= lenvalue:
                line += f.readline()
            mask, line = (line[1:lenvalue + 1], line[lenvalue + 1:])
        else:
            mask = None
        ending = cls.rule_ending_re.match(line)
        if not ending:
            raise UnknownMagicRuleFormat(repr(line))
        word, range = ending.groups()
        word = int(word) if word is not None else 1
        range = int(range) if range is not None else 1
        return (nest_depth, cls(start, value, mask, word, range))

    def maxlen(self):
        l = self.start + len(self.value) + self.range
        if self.also:
            return max(l, self.also.maxlen())
        return l

    def match(self, buffer):
        if self.match0(buffer):
            if self.also:
                return self.also.match(buffer)
            return True

    def match0(self, buffer):
        l = len(buffer)
        lenvalue = len(self.value)
        for o in range(self.range):
            s = self.start + o
            e = s + lenvalue
            if l < e:
                return False
            if self.mask:
                test = ''
                for i in range(lenvalue):
                    if PY3:
                        c = buffer[s + i] & self.mask[i]
                    else:
                        c = ord(buffer[s + i]) & ord(self.mask[i])
                    test += chr(c)
            else:
                test = buffer[s:e]
            if test == self.value:
                return True

    def __repr__(self):
        return 'MagicRule(start=%r, value=%r, mask=%r, word=%r, range=%r)' % (self.start, self.value, self.mask, self.word, self.range)