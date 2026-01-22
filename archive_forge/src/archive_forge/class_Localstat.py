import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Localstat(AFNICommand):
    """3dLocalstat - computes statistics at each voxel,
    based on a local neighborhood of that voxel.

    For complete details, see the `3dLocalstat Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dLocalstat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> localstat = afni.Localstat()
    >>> localstat.inputs.in_file = 'functional.nii'
    >>> localstat.inputs.mask_file = 'skeleton_mask.nii.gz'
    >>> localstat.inputs.neighborhood = ('SPHERE', 45)
    >>> localstat.inputs.stat = 'mean'
    >>> localstat.inputs.nonmask = True
    >>> localstat.inputs.outputtype = 'NIFTI_GZ'
    >>> localstat.cmdline
    "3dLocalstat -prefix functional_localstat.nii -mask skeleton_mask.nii.gz -nbhd 'SPHERE(45.0)' -use_nonmask -stat mean functional.nii"
    >>> res = localstat.run()  # doctest: +SKIP

    """
    _cmd = '3dLocalstat'
    input_spec = LocalstatInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'neighborhood' and value[0] == 'RECT':
            value = ('RECT', '%s,%s,%s' % value[1])
        if name == 'stat':
            value = ['perc:%s:%s:%s' % v[1] if len(v) == 2 else v for v in value]
        if name == 'reduce_grid' or name == 'reduce_restore_grid':
            if len(value) == 3:
                value = '%s %s %s' % value
        return super(Localstat, self)._format_arg(name, spec, value)