import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _get_codon_rec(pro, nucl, span_mode, gap_char, codon_table, complete_protein=False, max_score=10):
    """Generate codon alignment based on regular re match (PRIVATE).

    span_mode is a tuple returned by _check_corr. The first element
    is the span of a re search, and the second element is the mode
    for the match.

    mode
     - 0: direct match
     - 1: mismatch (no indels)
     - 2: frameshift

    """
    import re
    from Bio.Seq import Seq
    nucl_seq = nucl.seq.replace(gap_char, '')
    span = span_mode[0]
    mode = span_mode[1]
    aa2re = _get_aa_regex(codon_table)
    if mode in (0, 1):
        if len(pro.seq.replace(gap_char, '')) * 3 != span[1] - span[0]:
            raise ValueError(f'Protein Record {pro.id} and Nucleotide Record {nucl.id} do not match!')
        aa_num = 0
        codon_seq = CodonSeq()
        for aa in pro.seq:
            if aa == '-':
                codon_seq += '---'
            elif complete_protein and aa_num == 0:
                this_codon = nucl_seq[span[0]:span[0] + 3]
                if not re.search(_codons2re(codon_table.start_codons), str(this_codon.upper())):
                    max_score -= 1
                    warnings.warn(f'start codon of {pro.id} ({aa} {aa_num}) does not correspond to {nucl.id} ({this_codon})', BiopythonWarning)
                if max_score == 0:
                    raise RuntimeError(f'max_score reached for {nucl.id}! Please raise up the tolerance to get an alignment in anyway')
                codon_seq += this_codon
                aa_num += 1
            else:
                this_codon = nucl_seq[span[0] + 3 * aa_num:span[0] + 3 * (aa_num + 1)]
                if this_codon.upper().translate(table=codon_table) != aa:
                    max_score -= 1
                    warnings.warn('%s(%s %d) does not correspond to %s(%s)' % (pro.id, aa, aa_num, nucl.id, this_codon), BiopythonWarning)
                if max_score == 0:
                    raise RuntimeError(f'max_score reached for {nucl.id}! Please raise up the tolerance to get an alignment in anyway')
                codon_seq += this_codon
                aa_num += 1
        return SeqRecord(codon_seq, id=nucl.id)
    elif mode == 2:
        from collections import deque
        shift_pos = deque([])
        shift_start = []
        match = span_mode[2]
        m_groupdict = list(match.groupdict().keys())
        for i in m_groupdict:
            shift_pos.append(match.span(i))
            shift_start.append(match.start(i))
        rf_table = []
        i = match.start()
        while True:
            rf_table.append(i)
            i += 3
            if i in shift_start and m_groupdict[shift_start.index(i)].isupper():
                shift_index = shift_start.index(i)
                shift_val = 6 - (shift_pos[shift_index][1] - shift_pos[shift_index][0])
                rf_table.append(i)
                rf_table.append(i + 3 - shift_val)
                i = shift_pos[shift_index][1]
            elif i in shift_start and m_groupdict[shift_start.index(i)].islower():
                i = shift_pos[shift_start.index(i)][1]
            if i >= match.end():
                break
        codon_seq = CodonSeq()
        aa_num = 0
        for aa in pro.seq:
            if aa == '-':
                codon_seq += '---'
            elif complete_protein and aa_num == 0:
                this_codon = nucl_seq[rf_table[0]:rf_table[0] + 3]
                if not re.search(_codons2re(codon_table.start_codons), str(this_codon.upper())):
                    max_score -= 1
                    warnings.warn(f'start codon of {pro.id}({aa} {aa_num}) does not correspond to {nucl.id}({this_codon})', BiopythonWarning)
                    codon_seq += this_codon
                    aa_num += 1
            else:
                if aa_num < len(pro.seq.replace('-', '')) - 1 and rf_table[aa_num + 1] - rf_table[aa_num] - 3 < 0:
                    max_score -= 1
                    start = rf_table[aa_num]
                    end = start + (3 - shift_val)
                    ngap = shift_val
                    this_codon = nucl_seq[start:end] + '-' * ngap
                elif rf_table[aa_num] - rf_table[aa_num - 1] - 3 > 0:
                    max_score -= 1
                    start = rf_table[aa_num - 1] + 3
                    end = rf_table[aa_num]
                    ngap = 3 - (rf_table[aa_num] - rf_table[aa_num - 1] - 3)
                    this_codon = nucl_seq[start:end] + '-' * ngap + nucl_seq[rf_table[aa_num]:rf_table[aa_num] + 3]
                else:
                    start = rf_table[aa_num]
                    end = start + 3
                    this_codon = nucl_seq[start:end]
                    if this_codon.upper().translate(table=codon_table) != aa:
                        max_score -= 1
                        warnings.warn(f'Codon of {pro.id}({aa} {aa_num}) does not correspond to {nucl.id}({this_codon})', BiopythonWarning)
                if max_score == 0:
                    raise RuntimeError(f'max_score reached for {nucl.id}! Please raise up the tolerance to get an alignment in anyway')
                codon_seq += this_codon
                aa_num += 1
        codon_seq.rf_table = rf_table
        return SeqRecord(codon_seq, id=nucl.id)