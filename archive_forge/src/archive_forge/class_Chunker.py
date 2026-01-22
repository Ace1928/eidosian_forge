import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class Chunker(tokenize):
    """Base class for text chunking functions.

    A chunker is designed to chunk text into large blocks of tokens.  It
    has the same interface as a tokenizer but is for a different purpose.
    """
    pass