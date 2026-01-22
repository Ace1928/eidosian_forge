from Textco BioSoftware, Inc.
from struct import unpack
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
Start parsing the file, and return a SeqRecord generator.

        Note that a GCK file can only contain one sequence, so this
        iterator will always return a single record.
        