import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class BinaryStats(StatsCommand):
    """Binary statistical operations.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces import niftyseg
    >>> binary = niftyseg.BinaryStats()
    >>> binary.inputs.in_file = 'im1.nii'
    >>> # Test sa operation
    >>> binary_sa = copy.deepcopy(binary)
    >>> binary_sa.inputs.operation = 'sa'
    >>> binary_sa.inputs.operand_value = 2.0
    >>> binary_sa.cmdline
    'seg_stats im1.nii -sa 2.00000000'
    >>> binary_sa.run()  # doctest: +SKIP
    >>> # Test ncc operation
    >>> binary_ncc = copy.deepcopy(binary)
    >>> binary_ncc.inputs.operation = 'ncc'
    >>> binary_ncc.inputs.operand_file = 'im2.nii'
    >>> binary_ncc.cmdline
    'seg_stats im1.nii -ncc im2.nii'
    >>> binary_ncc.run()  # doctest: +SKIP
    >>> # Test Nl operation
    >>> binary_nl = copy.deepcopy(binary)
    >>> binary_nl.inputs.operation = 'Nl'
    >>> binary_nl.inputs.operand_file = 'output.csv'
    >>> binary_nl.cmdline
    'seg_stats im1.nii -Nl output.csv'
    >>> binary_nl.run()  # doctest: +SKIP

    """
    input_spec = BinaryStatsInput