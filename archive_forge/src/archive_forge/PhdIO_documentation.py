from the Biopython unit tests:
from typing import Iterator
from Bio.SeqRecord import SeqRecord
from Bio.Sequencing import Phd
from .QualityIO import _get_phred_quality
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from .Interfaces import _IOSource
Write a single Phd record to the file.