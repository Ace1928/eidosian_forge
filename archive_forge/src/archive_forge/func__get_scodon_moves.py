import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _get_scodon_moves(tmp_seq_blocks):
    """Get a dictionary of split codon locations relative to each fragment end (PRIVATE)."""
    scodon_moves = {'query': [], 'hit': []}
    for seq_type in scodon_moves:
        scoords = []
        for block in tmp_seq_blocks:
            m_start = re.search(_RE_SCODON_START, block[seq_type])
            m_end = re.search(_RE_SCODON_END, block[seq_type])
            if m_start:
                m_start = len(m_start.group(1))
                scoords.append((m_start, 0))
            else:
                scoords.append((0, 0))
            if m_end:
                m_end = len(m_end.group(1))
                scoords.append((0, m_end))
            else:
                scoords.append((0, 0))
        scodon_moves[seq_type] = scoords
    return scodon_moves