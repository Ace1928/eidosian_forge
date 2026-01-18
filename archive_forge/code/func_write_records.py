from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def write_records(self, records, maxcount=None):
    """Write records to the output file, and return the number of records.

        records - A list or iterator returning SeqRecord objects
        maxcount - The maximum number of records allowed by the
        file format, or None if there is no maximum.
        """
    count = 0
    if maxcount is None:
        for record in records:
            self.write_record(record)
            count += 1
    else:
        for record in records:
            if count == maxcount:
                if maxcount == 1:
                    raise ValueError('More than one sequence found')
                else:
                    raise ValueError('Number of sequences is larger than %d' % maxcount)
            self.write_record(record)
            count += 1
    return count