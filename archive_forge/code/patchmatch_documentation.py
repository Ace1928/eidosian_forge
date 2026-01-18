import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
Interface for executable seg_PatchMatch from NiftySeg platform.

    The database file is a text file and in each line we have a template
    file, a mask with the search region to consider and a file with the
    label to propagate.

    Input image, input mask, template images from database and masks from
    database must have the same 4D resolution (same number of XxYxZ voxels,
    modalities and/or time-points).
    Label files from database must have the same 3D resolution
    (XxYxZ voxels) than input image but can have different number of
    volumes than the input image allowing to propagate multiple labels
    in the same execution.

    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`_ |
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyseg
    >>> node = niftyseg.PatchMatch()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.mask_file = 'im2.nii'
    >>> node.inputs.database_file = 'db.xml'
    >>> node.cmdline
    'seg_PatchMatch -i im1.nii -m im2.nii -db db.xml -o im1_pm.nii.gz'

    