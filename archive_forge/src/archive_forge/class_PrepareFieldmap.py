import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class PrepareFieldmap(FSLCommand):
    """
    Interface for the fsl_prepare_fieldmap script (FSL 5.0)

    Prepares a fieldmap suitable for FEAT from SIEMENS data - saves output in
    rad/s format (e.g. ```fsl_prepare_fieldmap SIEMENS
    images_3_gre_field_mapping images_4_gre_field_mapping fmap_rads 2.65```).


    Examples
    --------

    >>> from nipype.interfaces.fsl import PrepareFieldmap
    >>> prepare = PrepareFieldmap()
    >>> prepare.inputs.in_phase = "phase.nii"
    >>> prepare.inputs.in_magnitude = "magnitude.nii"
    >>> prepare.inputs.output_type = "NIFTI_GZ"
    >>> prepare.cmdline # doctest: +ELLIPSIS
    'fsl_prepare_fieldmap SIEMENS phase.nii magnitude.nii .../phase_fslprepared.nii.gz 2.460000'
    >>> res = prepare.run() # doctest: +SKIP


    """
    _cmd = 'fsl_prepare_fieldmap'
    input_spec = PrepareFieldmapInputSpec
    output_spec = PrepareFieldmapOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if not isdefined(self.inputs.out_fieldmap):
            self.inputs.out_fieldmap = self._gen_fname(self.inputs.in_phase, suffix='_fslprepared')
        if not isdefined(self.inputs.nocheck) or not self.inputs.nocheck:
            skip += ['nocheck']
        return super(PrepareFieldmap, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_fieldmap'] = self.inputs.out_fieldmap
        return outputs

    def _run_interface(self, runtime):
        runtime = super(PrepareFieldmap, self)._run_interface(runtime)
        if runtime.returncode == 0:
            out_file = self.inputs.out_fieldmap
            im = nb.load(out_file)
            dumb_img = nb.Nifti1Image(np.zeros(im.shape), im.affine, im.header)
            out_nii = nb.funcs.concat_images((im, dumb_img))
            nb.save(out_nii, out_file)
        return runtime