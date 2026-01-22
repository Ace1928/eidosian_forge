import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Surface2VolTransform(FSCommand):
    """Use FreeSurfer mri_surf2vol to apply a transform.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import Surface2VolTransform
    >>> xfm2vol = Surface2VolTransform()
    >>> xfm2vol.inputs.source_file = 'lh.cope1.mgz'
    >>> xfm2vol.inputs.reg_file = 'register.mat'
    >>> xfm2vol.inputs.hemi = 'lh'
    >>> xfm2vol.inputs.template_file = 'cope1.nii.gz'
    >>> xfm2vol.inputs.subjects_dir = '.'
    >>> xfm2vol.cmdline
    'mri_surf2vol --hemi lh --volreg register.mat --surfval lh.cope1.mgz --sd . --template cope1.nii.gz --outvol lh.cope1_asVol.nii --vtxvol lh.cope1_asVol_vertex.nii'
    >>> res = xfm2vol.run()# doctest: +SKIP

    """
    _cmd = 'mri_surf2vol'
    input_spec = Surface2VolTransformInputSpec
    output_spec = Surface2VolTransformOutputSpec