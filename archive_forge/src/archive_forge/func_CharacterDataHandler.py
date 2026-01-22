import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError
import numpy as np
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
def CharacterDataHandler(self, data):
    """Collect character data chunks pending collation

        The parser breaks the data up into chunks of size depending on the
        buffer_size of the parser.  A large bit of character data, with
        standard parser buffer_size (such as 8K) can easily span many calls to
        this function.  We thus collect the chunks and process them when we
        hit start or end tags.
        """
    if self._char_blocks is None:
        self._char_blocks = []
    self._char_blocks.append(data)