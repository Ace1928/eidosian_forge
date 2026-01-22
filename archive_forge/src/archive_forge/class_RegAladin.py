import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegAladin(NiftyRegCommand):
    """Interface for executable reg_aladin from NiftyReg platform.

    Block Matching algorithm for symmetric global registration.
    Based on Modat et al., "Global image registration using
    asymmetric block-matching approach"
    J. Med. Img. 1(2) 024003, 2014, doi: 10.1117/1.JMI.1.2.024003

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegAladin()
    >>> node.inputs.ref_file = 'im1.nii'
    >>> node.inputs.flo_file = 'im2.nii'
    >>> node.inputs.rmask_file = 'mask.nii'
    >>> node.inputs.omp_core_val = 4
    >>> node.cmdline
    'reg_aladin -aff im2_aff.txt -flo im2.nii -omp 4 -ref im1.nii -res im2_res.nii.gz -rmask mask.nii'

    """
    _cmd = get_custom_path('reg_aladin')
    input_spec = RegAladinInputSpec
    output_spec = RegAladinOutputSpec

    def _list_outputs(self):
        outputs = super(RegAladin, self)._list_outputs()
        aff = os.path.abspath(outputs['aff_file'])
        flo = os.path.abspath(self.inputs.flo_file)
        outputs['avg_output'] = '%s %s' % (aff, flo)
        return outputs