import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class UnaryMaths(MathsCommand):
    """Unary mathematical operations.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces import niftyseg
    >>> unary = niftyseg.UnaryMaths()
    >>> unary.inputs.output_datatype = 'float'
    >>> unary.inputs.in_file = 'im1.nii'

    >>> # Test sqrt operation
    >>> unary_sqrt = copy.deepcopy(unary)
    >>> unary_sqrt.inputs.operation = 'sqrt'
    >>> unary_sqrt.cmdline
    'seg_maths im1.nii -sqrt -odt float im1_sqrt.nii'
    >>> unary_sqrt.run()  # doctest: +SKIP

    >>> # Test sqrt operation
    >>> unary_abs = copy.deepcopy(unary)
    >>> unary_abs.inputs.operation = 'abs'
    >>> unary_abs.cmdline
    'seg_maths im1.nii -abs -odt float im1_abs.nii'
    >>> unary_abs.run()  # doctest: +SKIP

    >>> # Test bin operation
    >>> unary_bin = copy.deepcopy(unary)
    >>> unary_bin.inputs.operation = 'bin'
    >>> unary_bin.cmdline
    'seg_maths im1.nii -bin -odt float im1_bin.nii'
    >>> unary_bin.run()  # doctest: +SKIP

    >>> # Test otsu operation
    >>> unary_otsu = copy.deepcopy(unary)
    >>> unary_otsu.inputs.operation = 'otsu'
    >>> unary_otsu.cmdline
    'seg_maths im1.nii -otsu -odt float im1_otsu.nii'
    >>> unary_otsu.run()  # doctest: +SKIP

    >>> # Test isnan operation
    >>> unary_isnan = copy.deepcopy(unary)
    >>> unary_isnan.inputs.operation = 'isnan'
    >>> unary_isnan.cmdline
    'seg_maths im1.nii -isnan -odt float im1_isnan.nii'
    >>> unary_isnan.run()  # doctest: +SKIP

    """
    input_spec = UnaryMathsInput