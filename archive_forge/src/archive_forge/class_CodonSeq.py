from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
class CodonSeq(Seq):
    """CodonSeq is designed to be within the SeqRecords of a CodonAlignment class.

    CodonSeq is useful as it allows the user to specify
    reading frame when translate CodonSeq

    CodonSeq also accepts codon style slice by calling
    get_codon() method.

    **Important:** Ungapped CodonSeq can be any length if you
    specify the rf_table. Gapped CodonSeq should be a
    multiple of three.

    >>> codonseq = CodonSeq("AAATTTGGGCCAAATTT", rf_table=(0,3,6,8,11,14))
    >>> print(codonseq.translate())
    KFGAKF

    test get_full_rf_table method

    >>> p = CodonSeq('AAATTTCCCGG-TGGGTTTAA', rf_table=(0, 3, 6, 9, 11, 14, 17))
    >>> full_rf_table = p.get_full_rf_table()
    >>> print(full_rf_table)
    [0, 3, 6, 9, 12, 15, 18]
    >>> print(p.translate(rf_table=full_rf_table, ungap_seq=False))
    KFPPWV*
    >>> p = CodonSeq('AAATTTCCCGGGAA-TTTTAA', rf_table=(0, 3, 6, 9, 14, 17))
    >>> print(p.get_full_rf_table())
    [0, 3, 6, 9, 12.0, 15, 18]
    >>> p = CodonSeq('AAA------------TAA', rf_table=(0, 3))
    >>> print(p.get_full_rf_table())
    [0, 3.0, 6.0, 9.0, 12.0, 15]

    """

    def __init__(self, data='', gap_char='-', rf_table=None):
        """Initialize the class."""
        Seq.__init__(self, data.upper())
        self.gap_char = gap_char
        if rf_table is None:
            length = len(self)
            if length % 3 != 0:
                raise ValueError('Sequence length is not a multiple of three (i.e. a whole number of codons)')
            self.rf_table = list(range(0, length - self.count(gap_char), 3))
        else:
            if not isinstance(rf_table, (tuple, list)):
                raise TypeError('rf_table should be a tuple or list object')
            if not all((isinstance(i, int) for i in rf_table)):
                raise TypeError('Elements in rf_table should be int that specify the codon positions of the sequence')
            self.rf_table = rf_table

    def get_codon(self, index):
        """Get the index codon from the sequence."""
        if len({i % 3 for i in self.rf_table}) != 1:
            raise RuntimeError('frameshift detected. CodonSeq object is not able to deal with codon sequence with frameshift. Please use normal slice option.')
        if isinstance(index, int):
            if index != -1:
                return str(self[index * 3:(index + 1) * 3])
            else:
                return str(self[index * 3:])
        else:
            aa_index = range(len(self) // 3)

            def cslice(p):
                aa_slice = aa_index[p]
                codon_slice = ''
                for i in aa_slice:
                    codon_slice += self[i * 3:i * 3 + 3]
                return str(codon_slice)
            codon_slice = cslice(index)
            return CodonSeq(codon_slice)

    def get_codon_num(self):
        """Return the number of codons in the CodonSeq."""
        return len(self.rf_table)

    def translate(self, codon_table=None, stop_symbol='*', rf_table=None, ungap_seq=True):
        """Translate the CodonSeq based on the reading frame in rf_table.

        It is possible for the user to specify
        a rf_table at this point. If you want to include
        gaps in the translated sequence, this is the only
        way. ungap_seq should be set to true for this
        purpose.
        """
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        amino_acids = []
        if ungap_seq:
            tr_seq = str(self).replace(self.gap_char, '')
        else:
            tr_seq = str(self)
        if rf_table is None:
            rf_table = self.rf_table
        p = -1
        for i in rf_table:
            if isinstance(i, float):
                amino_acids.append('-')
                continue
            elif '-' in tr_seq[i:i + 3]:
                if p == -1 or p - i == 3:
                    p = i
                    codon = tr_seq[i:i + 6].replace('-', '')[:3]
                elif p - i > 3:
                    codon = tr_seq[i:i + 3]
                    p = i
            else:
                codon = tr_seq[i:i + 3]
                p = i
            if codon in codon_table.stop_codons:
                amino_acids.append(stop_symbol)
                continue
            try:
                amino_acids.append(codon_table.forward_table[codon])
            except KeyError:
                raise RuntimeError(f'Unknown codon detected ({codon}). Did you forget to specify the ungap_seq argument?')
        return ''.join(amino_acids)

    def toSeq(self):
        """Convert DNA to seq object."""
        return Seq(str(self))

    def get_full_rf_table(self):
        """Return full rf_table of the CodonSeq records.

        A full rf_table is different from a normal rf_table in that
        it translate gaps in CodonSeq. It is helpful to construct
        alignment containing frameshift.
        """
        ungap_seq = str(self).replace('-', '')
        relative_pos = [self.rf_table[0]]
        for i in range(1, len(self.rf_table[1:]) + 1):
            relative_pos.append(self.rf_table[i] - self.rf_table[i - 1])
        full_rf_table = []
        codon_num = 0
        for i in range(0, len(self), 3):
            if self[i:i + 3] == self.gap_char * 3:
                full_rf_table.append(i + 0.0)
            elif relative_pos[codon_num] == 0:
                full_rf_table.append(i)
                codon_num += 1
            elif relative_pos[codon_num] in (-1, -2):
                gap_stat = 3 - self.count('-', i - 3, i)
                if gap_stat == 3:
                    full_rf_table.append(i + relative_pos[codon_num])
                elif gap_stat == 2:
                    full_rf_table.append(i + 1 + relative_pos[codon_num])
                elif gap_stat == 1:
                    full_rf_table.append(i + 2 + relative_pos[codon_num])
                codon_num += 1
            elif relative_pos[codon_num] > 0:
                full_rf_table.append(i + 0.0)
            try:
                this_len = 3 - self.count('-', i, i + 3)
                relative_pos[codon_num] -= this_len
            except Exception:
                pass
        return full_rf_table

    def full_translate(self, codon_table=None, stop_symbol='*'):
        """Apply full translation with gaps considered."""
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        full_rf_table = self.get_full_rf_table()
        return self.translate(codon_table=codon_table, stop_symbol=stop_symbol, rf_table=full_rf_table, ungap_seq=False)

    def ungap(self, gap='-'):
        """Return a copy of the sequence without the gap character(s)."""
        if len(gap) != 1 or not isinstance(gap, str):
            raise ValueError(f'Unexpected gap character, {gap!r}')
        return CodonSeq(str(self).replace(gap, ''), rf_table=self.rf_table)

    @classmethod
    def from_seq(cls, seq, rf_table=None):
        """Get codon sequence from sequence data."""
        if rf_table is None:
            return cls(str(seq))
        else:
            return cls(str(seq), rf_table=rf_table)