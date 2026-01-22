import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def aromaticity(self):
    """Calculate the aromaticity according to Lobry, 1994.

        Calculates the aromaticity value of a protein according to Lobry, 1994.
        It is simply the relative frequency of Phe+Trp+Tyr.
        """
    aromatic_aas = 'YWF'
    aa_percentages = self.get_amino_acids_percent()
    aromaticity = sum((aa_percentages[aa] for aa in aromatic_aas))
    return aromaticity