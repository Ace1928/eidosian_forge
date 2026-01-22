import re
from Bio.SearchIO._utils import read_forward, removesuffix
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def _parse_aln_block(self, hid, hsp_list):
    """Parse a HMMER3 HSP alignment block (PRIVATE)."""
    self.line = read_forward(self.handle)
    dom_counter = 0
    while True:
        if self.line.startswith('>>') or self.line.startswith('Internal pipeline'):
            return hsp_list
        assert self.line.startswith('  == domain %i' % (dom_counter + 1))
        frag = hsp_list[dom_counter][0]
        hmmseq = ''
        aliseq = ''
        annot = {}
        aln_prefix_len = None
        self.line = self.handle.readline()
        while True:
            regx = None
            regx = re.search(_HRE_ID_LINE, self.line)
            if regx:
                if aln_prefix_len is None:
                    aln_prefix_len = len(regx.group(1))
                else:
                    assert aln_prefix_len == len(regx.group(1))
                if len(hmmseq) == len(aliseq):
                    hmmseq += regx.group(2)
                elif len(hmmseq) > len(aliseq):
                    aliseq += regx.group(2)
                assert len(hmmseq) >= len(aliseq)
            elif self.line.startswith('  == domain') or self.line.startswith('>>') or self.line.startswith('Internal pipeline'):
                frag.aln_annotation = annot
                if self._meta.get('program') == 'hmmscan':
                    frag.hit = hmmseq
                    frag.query = aliseq
                elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                    frag.hit = aliseq
                    frag.query = hmmseq
                dom_counter += 1
                hmmseq = ''
                aliseq = ''
                annot = {}
                aln_prefix_len = None
                break
            elif len(hmmseq) == len(aliseq):
                regx = re.search(_HRE_ANNOT_LINE, self.line)
                if regx:
                    annot_name = regx.group(3)
                    if annot_name in annot:
                        annot[annot_name] += regx.group(2)
                    else:
                        annot[annot_name] = regx.group(2)
            elif aln_prefix_len is not None:
                similarity = removesuffix(removesuffix(self.line[aln_prefix_len:], '\n'), '\r')
                if 'similarity' not in annot:
                    annot['similarity'] = similarity
                else:
                    annot['similarity'] += similarity
            self.line = self.handle.readline()