import struct
from ..common.utils import struct_parse
from .sections import Section
def _matches_bloom(self, H1):
    """ Helper function to check if the given hash could be in the hash
            table by testing it against the bloom filter.
        """
    arch_bits = self.elffile.elfclass
    H2 = H1 >> self.params['bloom_shift']
    word_idx = int(H1 / arch_bits) % self.params['bloom_size']
    BITMASK = 1 << H1 % arch_bits | 1 << H2 % arch_bits
    return self.params['bloom'][word_idx] & BITMASK == BITMASK