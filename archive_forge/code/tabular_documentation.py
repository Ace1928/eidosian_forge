import re
import enum
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
Parse a CIGAR string and return alignment coordinates.

        A CIGAR string, as defined by the SAM Sequence Alignment/Map format,
        describes a sequence alignment as a series of lengths and operation
        (alignment/insertion/deletion) codes.
        