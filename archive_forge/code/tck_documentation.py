import os
import warnings
from contextlib import suppress
import numpy as np
from nibabel.openers import Opener
from .array_sequence import ArraySequence
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, DataWarning, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
Gets a formatted string of the header of a TCK file.

        Returns
        -------
        info : string
            Header information relevant to the TCK format.
        