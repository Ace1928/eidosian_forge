import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class BinaryMathsInteger(MathsCommand):
    """Integer mathematical operations.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces.niftyseg import BinaryMathsInteger
    >>> binaryi = BinaryMathsInteger()
    >>> binaryi.inputs.in_file = 'im1.nii'
    >>> binaryi.inputs.output_datatype = 'float'
    >>> # Test dil operation
    >>> binaryi_dil = copy.deepcopy(binaryi)
    >>> binaryi_dil.inputs.operation = 'dil'
    >>> binaryi_dil.inputs.operand_value = 2
    >>> binaryi_dil.cmdline
    'seg_maths im1.nii -dil 2 -odt float im1_dil.nii'
    >>> binaryi_dil.run()  # doctest: +SKIP
    >>> # Test dil operation
    >>> binaryi_ero = copy.deepcopy(binaryi)
    >>> binaryi_ero.inputs.operation = 'ero'
    >>> binaryi_ero.inputs.operand_value = 1
    >>> binaryi_ero.cmdline
    'seg_maths im1.nii -ero 1 -odt float im1_ero.nii'
    >>> binaryi_ero.run()  # doctest: +SKIP
    >>> # Test pad operation
    >>> binaryi_pad = copy.deepcopy(binaryi)
    >>> binaryi_pad.inputs.operation = 'pad'
    >>> binaryi_pad.inputs.operand_value = 4
    >>> binaryi_pad.cmdline
    'seg_maths im1.nii -pad 4 -odt float im1_pad.nii'
    >>> binaryi_pad.run()  # doctest: +SKIP

    """
    input_spec = BinaryMathsInputInteger