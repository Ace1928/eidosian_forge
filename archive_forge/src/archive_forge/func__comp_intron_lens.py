import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _comp_intron_lens(seq_type, inter_blocks, raw_inter_lens):
    """Return the length of introns between fragments (PRIVATE)."""
    opp_type = 'hit' if seq_type == 'query' else 'query'
    has_intron_after = ['Intron' in x[seq_type] for x in inter_blocks]
    assert len(has_intron_after) == len(raw_inter_lens)
    inter_lens = []
    for flag, parsed_len in zip(has_intron_after, raw_inter_lens):
        if flag:
            if all(parsed_len[:2]):
                intron_len = int(parsed_len[0]) if opp_type == 'query' else int(parsed_len[1])
            elif parsed_len[2]:
                intron_len = int(parsed_len[2])
            else:
                raise ValueError('Unexpected intron parsing result: %r' % parsed_len)
        else:
            intron_len = 0
        inter_lens.append(intron_len)
    return inter_lens