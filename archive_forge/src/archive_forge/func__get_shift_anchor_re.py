import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _get_shift_anchor_re(sh_anc, sh_nuc, shift_val, aa2re, anchor_len, shift_id_pos):
    """Find a regular expression matching a potentially shifted anchor (PRIVATE).

    Arguments:
     - sh_anc    - shifted anchor sequence
     - sh_nuc    - potentially corresponding nucleotide sequence
       of sh_anc
     - shift_val - 1 or 2 indicates forward frame shift, whereas
       3*anchor_len-1 or 3*anchor_len-2 indicates
       backward shift
     - aa2re     - aa to codon re dict
     - anchor_len - length of the anchor
     - shift_id_pos - specify current shift name we are at

    """
    import re
    shift_id = [chr(i) for i in range(97, 107)]
    if 0 < shift_val < 3 * anchor_len - 2:
        for j in range(len(sh_anc)):
            qcodon = '^'
            for k, aa in enumerate(sh_anc):
                if k == j:
                    qcodon += aa2re[aa] + '(?P<' + shift_id[shift_id_pos] + '>..*)'
                else:
                    qcodon += aa2re[aa]
            qcodon += '$'
            match = re.search(qcodon, sh_nuc)
            if match:
                qcodon = qcodon.replace('^', '').replace('$', '')
                shift_id_pos += 1
                return (qcodon, shift_id_pos)
        if not match:
            return (-1, shift_id_pos)
    elif shift_val in (3 * anchor_len - 1, 3 * anchor_len - 2):
        shift_val = 3 * anchor_len - shift_val
        for j in range(1, len(sh_anc)):
            qcodon = '^'
            for k, aa in enumerate(sh_anc):
                if k == j - 1:
                    pass
                elif k == j:
                    qcodon += _merge_aa2re(sh_anc[j - 1], sh_anc[j], shift_val, aa2re, shift_id[shift_id_pos].upper())
                else:
                    qcodon += aa2re[aa]
            qcodon += '$'
            match = re.search(qcodon, sh_nuc)
            if match:
                qcodon = qcodon.replace('^', '').replace('$', '')
                shift_id_pos += 1
                return (qcodon, shift_id_pos)
        if not match:
            return (-1, shift_id_pos)