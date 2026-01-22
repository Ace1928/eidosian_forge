import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
class CodonAdaptationIndex(dict):
    """A codon adaptation index (CAI) implementation.

    Implements the codon adaptation index (CAI) described by Sharp and
    Li (Nucleic Acids Res. 1987 Feb 11;15(3):1281-95).
    """

    def __init__(self, sequences, table=standard_dna_table):
        """Generate a codon adaptiveness table from the coding DNA sequences.

        This calculates the relative adaptiveness of each codon (w_ij) as
        defined by Sharp & Li (Nucleic Acids Research 15(3): 1281-1295 (1987))
        from the provided codon DNA sequences.

        Arguments:
         - sequences: An iterable over DNA sequences, which may be plain
                      strings, Seq objects, MutableSeq objects, or SeqRecord
                      objects.
         - table:     A Bio.Data.CodonTable.CodonTable object defining the
                      genetic code. By default, the standard genetic code is
                      used.
        """
        self._table = table
        codons = {aminoacid: [] for aminoacid in table.protein_alphabet}
        for codon, aminoacid in table.forward_table.items():
            codons[aminoacid].append(codon)
        synonymous_codons = tuple(list(codons.values()) + [table.stop_codons])
        counts = {c1 + c2 + c3: 0 for c1 in 'ACGT' for c2 in 'ACGT' for c3 in 'ACGT'}
        self.update(counts)
        for sequence in sequences:
            try:
                name = sequence.id
                sequence = sequence.seq
            except AttributeError:
                name = None
            sequence = sequence.upper()
            for i in range(0, len(sequence), 3):
                codon = sequence[i:i + 3]
                try:
                    counts[codon] += 1
                except KeyError:
                    if name is None:
                        message = f"illegal codon '{codon}'"
                    else:
                        message = f"illegal codon '{codon}' in gene {name}"
                    raise ValueError(message) from None
        for codon, count in counts.items():
            if count == 0:
                counts[codon] = 0.5
        for codons in synonymous_codons:
            denominator = max((counts[codon] for codon in codons))
            for codon in codons:
                self[codon] = counts[codon] / denominator

    def calculate(self, sequence):
        """Calculate and return the CAI (float) for the provided DNA sequence."""
        cai_value, cai_length = (0, 0)
        try:
            sequence = sequence.seq
        except AttributeError:
            pass
        sequence = sequence.upper()
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i + 3]
            if codon in ['ATG', 'TGG']:
                continue
            try:
                cai_value += log(self[codon])
            except KeyError:
                if codon in ['TGA', 'TAA', 'TAG']:
                    continue
                raise TypeError(f'illegal codon in sequence: {codon}') from None
            else:
                cai_length += 1
        return exp(cai_value / cai_length)

    def optimize(self, sequence, seq_type='DNA', strict=True):
        """Return a new DNA sequence with preferred codons only.

        Uses the codon adaptiveness table defined by the CodonAdaptationIndex
        object to generate DNA sequences with only preferred codons.
        May be useful when designing DNA sequences for transgenic protein
        expression or codon-optimized proteins like fluorophores.

        Arguments:
            - sequence: DNA, RNA, or protein sequence to codon-optimize.
                        Supplied as a str, Seq, or SeqRecord object.
            - seq_type: String specifying type of sequence provided.
                        Options are "DNA", "RNA", and "protein". Default is "DNA".
            - strict:   Determines whether an exception should be raised when
                        two codons are equally prefered for a given amino acid.
        Returns:
            Seq object with DNA encoding the same protein as the sequence argument,
            but using only preferred codons as defined by the codon adaptation index.
            If multiple codons are equally preferred, a warning is issued
            and one codon is chosen for use in the optimzed sequence.
        """
        try:
            sequence = sequence.seq
        except AttributeError:
            pass
        seq = sequence.upper()
        pref_codons = {}
        for codon, aminoacid in self._table.forward_table.items():
            if self[codon] == 1.0:
                if aminoacid in pref_codons:
                    msg = f'{pref_codons[aminoacid]} and {codon} are equally preferred.'
                    if strict:
                        raise ValueError(msg)
                pref_codons[aminoacid] = codon
        for codon in self._table.stop_codons:
            if self[codon] == 1.0:
                pref_codons['*'] = codon
        if seq_type == 'DNA' or seq_type == 'RNA':
            aa_seq = translate(seq)
        elif seq_type == 'protein':
            aa_seq = seq
        else:
            raise ValueError(f'Allowed seq_types are DNA, RNA or protein, not {seq_type!r}')
        try:
            optimized = ''.join((pref_codons[aa] for aa in aa_seq))
        except KeyError as ex:
            raise KeyError(f'Unrecognized amino acid: {ex}') from None
        return Seq(optimized)

    def __str__(self):
        lines = []
        for codon, value in self.items():
            line = f'{codon}\t{value:.3f}'
            lines.append(line)
        return '\n'.join(lines) + '\n'