from Bio.Data import IUPACData
from typing import Dict, List, Optional
class NCBICodonTableDNA(NCBICodonTable):
    """Codon table for unambiguous DNA sequences."""
    nucleotide_alphabet = IUPACData.unambiguous_dna_letters