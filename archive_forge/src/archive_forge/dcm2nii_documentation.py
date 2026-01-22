import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
Uses Chris Rorden's dcm2niix to convert dicom files

    Examples
    ========

    >>> from nipype.interfaces.dcm2nii import Dcm2niix
    >>> converter = Dcm2niix()
    >>> converter.inputs.source_dir = 'dicomdir'
    >>> converter.inputs.compression = 5
    >>> converter.inputs.output_dir = 'ds005'
    >>> converter.cmdline
    'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n dicomdir'
    >>> converter.run() # doctest: +SKIP

    In the example below, we note that the current version of dcm2niix
    converts any files in the directory containing the files in the list. We
    also do not support nested filenames with this option. **Thus all files
    must have a common root directory.**

    >>> converter = Dcm2niix()
    >>> converter.inputs.source_names = ['functional_1.dcm', 'functional_2.dcm']
    >>> converter.inputs.compression = 5
    >>> converter.inputs.output_dir = 'ds005'
    >>> converter.cmdline
    'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n .'
    >>> converter.run() # doctest: +SKIP
    