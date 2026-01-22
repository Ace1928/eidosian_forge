import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _merge_aa2re(aa1, aa2, shift_val, aa2re, reid):
    """Merge two amino acids based on detected frame shift value (PRIVATE)."""

    def get_aa_from_codonre(re_aa):
        aas = []
        m = 0
        for i in re_aa:
            if i == '[':
                m = -1
                aas.append('')
            elif i == ']':
                m = 0
                continue
            elif m == -1:
                aas[-1] = aas[-1] + i
            elif m == 0:
                aas.append(i)
        return aas
    scodon = list(map(get_aa_from_codonre, (aa2re[aa1], aa2re[aa2])))
    if shift_val == 1:
        intersect = ''.join(set(scodon[0][2]) & set(scodon[1][0]))
        scodonre = '(?P<' + reid + '>'
        scodonre += '[' + scodon[0][0] + ']' + '[' + scodon[0][1] + ']' + '[' + intersect + ']' + '[' + scodon[1][1] + ']' + '[' + scodon[1][2] + ']'
    elif shift_val == 2:
        intersect1 = ''.join(set(scodon[0][1]) & set(scodon[1][0]))
        intersect2 = ''.join(set(scodon[0][2]) & set(scodon[1][1]))
        scodonre = '(?P<' + reid + '>'
        scodonre += '[' + scodon[0][0] + ']' + '[' + intersect1 + ']' + '[' + intersect2 + ']' + '[' + scodon[1][2] + ']'
    scodonre += ')'
    return scodonre