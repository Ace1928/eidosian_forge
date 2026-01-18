from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def write_footer(self):
    """Write the file footer to the output file."""