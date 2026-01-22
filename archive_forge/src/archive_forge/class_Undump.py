import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Undump(AFNICommand):
    """3dUndump - Assembles a 3D dataset from an ASCII list of coordinates and
    (optionally) values.

     The input file(s) are ASCII files, with one voxel specification per
     line.  A voxel specification is 3 numbers (-ijk or -xyz coordinates),
     with an optional 4th number giving the voxel value.  For example:

     1 2 3
     3 2 1 5
     5.3 6.2 3.7
     // this line illustrates a comment

     The first line puts a voxel (with value given by '-dval') at point
     (1,2,3).  The second line puts a voxel (with value 5) at point (3,2,1).
     The third line puts a voxel (with value given by '-dval') at point
     (5.3,6.2,3.7).  If -ijk is in effect, and fractional coordinates
     are given, they will be rounded to the nearest integers; for example,
     the third line would be equivalent to (i,j,k) = (5,6,4).


    For complete details, see the `3dUndump Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dUndump.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> unndump = afni.Undump()
    >>> unndump.inputs.in_file = 'structural.nii'
    >>> unndump.inputs.out_file = 'structural_undumped.nii'
    >>> unndump.cmdline
    '3dUndump -prefix structural_undumped.nii -master structural.nii'
    >>> res = unndump.run()  # doctest: +SKIP

    """
    _cmd = '3dUndump'
    input_spec = UndumpInputSpec
    output_spec = UndumpOutputSpec