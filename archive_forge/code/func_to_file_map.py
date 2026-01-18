from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
def to_file_map(self, file_map=None, enc='utf-8', *, mode='strict'):
    """Save the current image to the specified file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        Returns
        -------
        None
        """
    if file_map is None:
        file_map = self.file_map
    with file_map['image'].get_prepare_fileobj('wb') as f:
        f.write(self.to_xml(enc=enc, mode=mode))