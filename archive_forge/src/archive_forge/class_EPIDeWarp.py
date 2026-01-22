import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EPIDeWarp(FSLCommand):
    """
    Wraps the unwarping script `epidewarp.fsl
    <http://surfer.nmr.mgh.harvard.edu/fswiki/epidewarp.fsl>`_.

    .. warning:: deprecated in FSL, please use
      :func:`niflow.nipype1.workflows.dmri.preprocess.epi.sdc_fmb` instead.

    Examples
    --------

    >>> from nipype.interfaces.fsl import EPIDeWarp
    >>> dewarp = EPIDeWarp()
    >>> dewarp.inputs.epi_file = "functional.nii"
    >>> dewarp.inputs.mag_file = "magnitude.nii"
    >>> dewarp.inputs.dph_file = "phase.nii"
    >>> dewarp.inputs.output_type = "NIFTI_GZ"
    >>> dewarp.cmdline # doctest: +ELLIPSIS
    'epidewarp.fsl --mag magnitude.nii --dph phase.nii --epi functional.nii --esp 0.58 --exfdw .../exfdw.nii.gz --nocleanup --sigma 2 --tediff 2.46 --tmpdir .../temp --vsm .../vsm.nii.gz'
    >>> res = dewarp.run() # doctest: +SKIP


    """
    _cmd = 'epidewarp.fsl'
    input_spec = EPIDeWarpInputSpec
    output_spec = EPIDeWarpOutputSpec

    def __init__(self, **inputs):
        warnings.warn('Deprecated: Please use niflow.nipype1.workflows.dmri.preprocess.epi.sdc_fmb instead', DeprecationWarning)
        return super(EPIDeWarp, self).__init__(**inputs)

    def _run_interface(self, runtime):
        runtime = super(EPIDeWarp, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _gen_filename(self, name):
        if name == 'exfdw':
            if isdefined(self.inputs.exf_file):
                return self._gen_fname(self.inputs.exf_file, suffix='_exfdw')
            else:
                return self._gen_fname('exfdw')
        if name == 'epidw':
            if isdefined(self.inputs.epi_file):
                return self._gen_fname(self.inputs.epi_file, suffix='_epidw')
        if name == 'vsm':
            return self._gen_fname('vsm')
        if name == 'tmpdir':
            return os.path.join(os.getcwd(), 'temp')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.exfdw):
            outputs['exfdw'] = self._gen_filename('exfdw')
        else:
            outputs['exfdw'] = self.inputs.exfdw
        if isdefined(self.inputs.epi_file):
            if isdefined(self.inputs.epidw):
                outputs['unwarped_file'] = self.inputs.epidw
            else:
                outputs['unwarped_file'] = self._gen_filename('epidw')
        if not isdefined(self.inputs.vsm):
            outputs['vsm_file'] = self._gen_filename('vsm')
        else:
            outputs['vsm_file'] = self._gen_fname(self.inputs.vsm)
        if not isdefined(self.inputs.tmpdir):
            outputs['exf_mask'] = self._gen_fname(cwd=self._gen_filename('tmpdir'), basename='maskexf')
        else:
            outputs['exf_mask'] = self._gen_fname(cwd=self.inputs.tmpdir, basename='maskexf')
        return outputs