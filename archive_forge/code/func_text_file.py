from itertools import chain
from .errors import BinaryFile
from .iterablefile import IterableFile
from .osutils import file_iterator
def text_file(input):
    """Produce a file iterator that is guaranteed to be text, without seeking.
    BinaryFile is raised if the file contains a NUL in the first 1024 bytes.
    """
    first_chunk = input.read(1024)
    if b'\x00' in first_chunk:
        raise BinaryFile()
    return IterableFile(chain((first_chunk,), file_iterator(input)))