import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class TupleMaths(MathsCommand):
    """Mathematical operations on tuples.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces import niftyseg
    >>> tuple = niftyseg.TupleMaths()
    >>> tuple.inputs.in_file = 'im1.nii'
    >>> tuple.inputs.output_datatype = 'float'

    >>> # Test lncc operation
    >>> tuple_lncc = copy.deepcopy(tuple)
    >>> tuple_lncc.inputs.operation = 'lncc'
    >>> tuple_lncc.inputs.operand_file1 = 'im2.nii'
    >>> tuple_lncc.inputs.operand_value2 = 2.0
    >>> tuple_lncc.cmdline
    'seg_maths im1.nii -lncc im2.nii 2.00000000 -odt float im1_lncc.nii'
    >>> tuple_lncc.run()  # doctest: +SKIP

    >>> # Test lssd operation
    >>> tuple_lssd = copy.deepcopy(tuple)
    >>> tuple_lssd.inputs.operation = 'lssd'
    >>> tuple_lssd.inputs.operand_file1 = 'im2.nii'
    >>> tuple_lssd.inputs.operand_value2 = 1.0
    >>> tuple_lssd.cmdline
    'seg_maths im1.nii -lssd im2.nii 1.00000000 -odt float im1_lssd.nii'
    >>> tuple_lssd.run()  # doctest: +SKIP

    >>> # Test lltsnorm operation
    >>> tuple_lltsnorm = copy.deepcopy(tuple)
    >>> tuple_lltsnorm.inputs.operation = 'lltsnorm'
    >>> tuple_lltsnorm.inputs.operand_file1 = 'im2.nii'
    >>> tuple_lltsnorm.inputs.operand_value2 = 0.01
    >>> tuple_lltsnorm.cmdline
    'seg_maths im1.nii -lltsnorm im2.nii 0.01000000 -odt float im1_lltsnorm.nii'
    >>> tuple_lltsnorm.run()  # doctest: +SKIP

    """
    input_spec = TupleMathsInput