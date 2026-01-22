import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
Scan over and parse GenBank LOCUS line (PRIVATE).

        This must cope with several variants, primarily the old and new column
        based standards from GenBank. Additionally EnsEMBL produces GenBank
        files where the LOCUS line is space separated rather that following
        the column based layout.

        We also try to cope with GenBank like files with partial LOCUS lines.

        As of release 229.0, the columns are no longer strictly in a given
        position. See GenBank format release notes:

            "Historically, the LOCUS line has had a fixed length and its
            elements have been presented at specific column positions...
            But with the anticipated increases in the lengths of accession
            numbers, and the advent of sequences that are gigabases long,
            maintaining the column positions will not always be possible and
            the overall length of the LOCUS line could exceed 79 characters."

        