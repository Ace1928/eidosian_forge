from Bio.Data import IUPACData
from typing import Dict, List, Optional
class NCBICodonTableRNA(NCBICodonTable):
    """Codon table for unambiguous RNA sequences."""
    nucleotide_alphabet = IUPACData.unambiguous_rna_letters