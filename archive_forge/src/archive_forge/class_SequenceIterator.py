from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class SequenceIterator(ABC, Generic[AnyStr]):
    """Base class for building SeqRecord iterators.

    You should write a parse method that returns a SeqRecord generator.  You
    may wish to redefine the __init__ method as well.
    """

    def __init__(self, source: _IOSource, alphabet: None=None, mode: str='t', fmt: Optional[str]=None) -> None:
        """Create a SequenceIterator object.

        Arguments:
        - source - input file stream, or path to input file
        - alphabet - no longer used, should be None

        This method MAY be overridden by any subclass.

        Note when subclassing:
        - there should be a single non-optional argument, the source.
        - you do not have to require an alphabet.
        - you can add additional optional arguments.
        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        if isinstance(source, _PathLikeTypes):
            self.stream = open(source, 'r' + mode)
            self.should_close_stream = True
        else:
            if mode == 't':
                if source.read(0) != '':
                    raise StreamModeError(f'{fmt} files must be opened in text mode.') from None
            elif mode == 'b':
                if source.read(0) != b'':
                    raise StreamModeError(f'{fmt} files must be opened in binary mode.') from None
            else:
                raise ValueError(f"Unknown mode '{mode}'") from None
            self.stream = source
            self.should_close_stream = False
        try:
            self.records = self.parse(self.stream)
        except Exception:
            if self.should_close_stream:
                self.stream.close()
            raise

    def __next__(self):
        """Return the next entry."""
        try:
            return next(self.records)
        except Exception:
            if self.should_close_stream:
                self.stream.close()
            raise

    def __iter__(self):
        """Iterate over the entries as a SeqRecord objects.

        Example usage for Fasta files::

            with open("example.fasta","r") as myFile:
                myFastaReader = FastaIterator(myFile)
                for record in myFastaReader:
                    print(record.id)
                    print(record.seq)

        This method SHOULD NOT be overridden by any subclass. It should be
        left as is, which will call the subclass implementation of __next__
        to actually parse the file.
        """
        return self

    @abstractmethod
    def parse(self, handle: IO[AnyStr]) -> Iterator[SeqRecord]:
        """Start parsing the file, and return a SeqRecord iterator."""