from Bio import SeqFeature
from Bio import SwissProt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
Break up a Swiss-Prot/UniProt file into SeqRecord objects.

    Argument source is a file-like object or a path to a file.

    Every section from the ID line to the terminating // becomes
    a single SeqRecord with associated annotation and features.

    This parser is for the flat file "swiss" format as used by:
     - Swiss-Prot aka SwissProt
     - TrEMBL
     - UniProtKB aka UniProt Knowledgebase

    For consistency with BioPerl and EMBOSS we call this the "swiss"
    format. See also the SeqIO support for "uniprot-xml" format.

    Rather than calling it directly, you are expected to use this
    parser via Bio.SeqIO.parse(..., format="swiss") instead.
    