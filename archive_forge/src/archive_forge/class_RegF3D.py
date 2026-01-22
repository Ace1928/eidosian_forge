import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegF3D(NiftyRegCommand):
    """Interface for executable reg_f3d from NiftyReg platform.

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

    """
    _cmd = get_custom_path('reg_f3d')
    input_spec = RegF3DInputSpec
    output_spec = RegF3DOutputSpec

    @staticmethod
    def _remove_extension(in_file):
        dn, bn, _ = split_filename(in_file)
        return os.path.join(dn, bn)

    def _list_outputs(self):
        outputs = super(RegF3D, self)._list_outputs()
        if self.inputs.vel_flag is True:
            res_name = self._remove_extension(outputs['res_file'])
            cpp_name = self._remove_extension(outputs['cpp_file'])
            outputs['invres_file'] = '%s_backward.nii.gz' % res_name
            outputs['invcpp_file'] = '%s_backward.nii.gz' % cpp_name
        if self.inputs.vel_flag is True and isdefined(self.inputs.aff_file):
            cpp_file = os.path.abspath(outputs['cpp_file'])
            flo_file = os.path.abspath(self.inputs.flo_file)
            outputs['avg_output'] = '%s %s %s' % (self.inputs.aff_file, cpp_file, flo_file)
        else:
            cpp_file = os.path.abspath(outputs['cpp_file'])
            flo_file = os.path.abspath(self.inputs.flo_file)
            outputs['avg_output'] = '%s %s' % (cpp_file, flo_file)
        return outputs