import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
Interface for executable reg_f3d from NiftyReg platform.

    Fast Free-Form Deformation (F3D) algorithm for non-rigid registration.
    Initially based on Modat et al., "Fast Free-Form Deformation using
    graphics processing units", CMPB, 2010

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegF3D()
    >>> node.inputs.ref_file = 'im1.nii'
    >>> node.inputs.flo_file = 'im2.nii'
    >>> node.inputs.rmask_file = 'mask.nii'
    >>> node.inputs.omp_core_val = 4
    >>> node.cmdline
    'reg_f3d -cpp im2_cpp.nii.gz -flo im2.nii -omp 4 -ref im1.nii -res im2_res.nii.gz -rmask mask.nii'

    