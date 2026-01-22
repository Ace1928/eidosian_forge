import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
class PSSM:
    """Represent a position specific score matrix.

    This class is meant to make it easy to access the info within a PSSM
    and also make it easy to print out the information in a nice table.

    Let's say you had an alignment like this::

        GTATC
        AT--C
        CTGTC

    The position specific score matrix (when printed) looks like::

          G A T C
        G 1 1 0 1
        T 0 0 3 0
        A 1 1 0 0
        T 0 0 2 0
        C 0 0 0 3

    You can access a single element of the PSSM using the following::

        your_pssm[sequence_number][residue_count_name]

    For instance, to get the 'T' residue for the second element in the
    above alignment you would need to do:

    your_pssm[1]['T']
    """

    def __init__(self, pssm):
        """Initialize with pssm data to represent.

        The pssm passed should be a list with the following structure:

        list[0] - The letter of the residue being represented (for instance,
        from the example above, the first few list[0]s would be GTAT...
        list[1] - A dictionary with the letter substitutions and counts.
        """
        warnings.warn("The `PSSM` class is deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and then create a Motif object. For example, for a multiple sequence alignment `msa` of DNA nucleotides, you would do: \n>>> alignment = msa.alignment\n>>> from Bio.motifs import Motif\n>>> motif = Motif('ACGT', alignment)\n>>> counts = motif.counts\n\nThe `counts` object contains the same information as the PSSM returned by `pos_specific_score_matrix`, but note that the indices are reversed:\n\n>>> counts[letter][i] == pssm[index][letter]\nTrue\n\nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.", BiopythonDeprecationWarning)
        self.pssm = pssm

    def __getitem__(self, pos):
        return self.pssm[pos][1]

    def __str__(self):
        out = ' '
        all_residues = sorted(self.pssm[0][1])
        for res in all_residues:
            out += '   %s' % res
        out += '\n'
        for item in self.pssm:
            out += '%s ' % item[0]
            for res in all_residues:
                out += ' %.1f' % item[1][res]
            out += '\n'
        return out

    def get_residue(self, pos):
        """Return the residue letter at the specified position."""
        return self.pssm[pos][0]