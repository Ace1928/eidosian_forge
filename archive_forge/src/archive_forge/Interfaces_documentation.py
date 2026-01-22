from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
Write a complete file with the records, and return the number of records.

        records - A list or iterator returning SeqRecord objects
        