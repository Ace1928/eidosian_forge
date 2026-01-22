import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _adjust_output(self, hsp, elem, attr):
    """Adjust output to mimic native BLAST+ XML as much as possible (PRIVATE)."""
    if attr in ('query_start', 'query_end', 'hit_start', 'hit_end', 'pattern_start', 'pattern_end'):
        content = getattr(hsp, attr) + 1
        if '_start' in attr:
            content = getattr(hsp, attr) + 1
        else:
            content = getattr(hsp, attr)
        if hsp.query_frame != 0 and hsp.hit_frame < 0:
            if attr == 'hit_start':
                content = getattr(hsp, 'hit_end')
            elif attr == 'hit_end':
                content = getattr(hsp, 'hit_start') + 1
    elif elem in ('Hsp_hseq', 'Hsp_qseq'):
        content = str(getattr(hsp, attr).seq)
    elif elem == 'Hsp_midline':
        content = hsp.aln_annotation['similarity']
    elif elem in ('Hsp_evalue', 'Hsp_bit-score'):
        content = '%.*g' % (6, getattr(hsp, attr))
    else:
        content = getattr(hsp, attr)
    return content