from __future__ import absolute_import
import math
import struct
import dns.inet
class GenericOption(Option):
    """Generic Option Class

    This class is used for EDNS option types for which we have no better
    implementation.
    """

    def __init__(self, otype, data):
        super(GenericOption, self).__init__(otype)
        self.data = data

    def to_wire(self, file):
        file.write(self.data)

    def to_text(self):
        return 'Generic %d' % self.otype

    @classmethod
    def from_wire(cls, otype, wire, current, olen):
        return cls(otype, wire[current:current + olen])

    def _cmp(self, other):
        if self.data == other.data:
            return 0
        if self.data > other.data:
            return 1
        return -1