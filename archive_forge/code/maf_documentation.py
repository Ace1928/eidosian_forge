import shlex
import itertools
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
Return a string with a single alignment formatted as a MAF block.