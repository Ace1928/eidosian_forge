import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIPreproc(MRTrix3Base):
    """
    Perform diffusion image pre-processing using FSL's eddy tool; including inhomogeneity distortion correction using FSL's topup tool if possible

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/dwifslpreproc.html>

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> preproc = mrt.DWIPreproc()
    >>> preproc.inputs.in_file = 'dwi.mif'
    >>> preproc.inputs.rpe_options = 'none'
    >>> preproc.inputs.out_file = "preproc.mif"
    >>> preproc.inputs.eddy_options = '--slm=linear --repol'     # linear second level model and replace outliers
    >>> preproc.inputs.out_grad_mrtrix = "grad.b"    # export final gradient table in MRtrix format
    >>> preproc.inputs.ro_time = 0.165240   # 'TotalReadoutTime' in BIDS JSON metadata files
    >>> preproc.inputs.pe_dir = 'j'     # 'PhaseEncodingDirection' in BIDS JSON metadata files
    >>> preproc.cmdline
    'dwifslpreproc dwi.mif preproc.mif -rpe_none -eddy_options "--slm=linear --repol" -export_grad_mrtrix grad.b -pe_dir j -readout_time 0.165240'
    >>> preproc.run()                             # doctest: +SKIP
    """
    _cmd = 'dwifslpreproc'
    input_spec = DWIPreprocInputSpec
    output_spec = DWIPreprocOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        if self.inputs.export_grad_mrtrix:
            outputs['out_grad_mrtrix'] = op.abspath(self.inputs.out_grad_mrtrix)
        if self.inputs.export_grad_fsl:
            outputs['out_fsl_bvec'] = op.abspath(self.inputs.out_grad_fsl[0])
            outputs['out_fsl_bval'] = op.abspath(self.inputs.out_grad_fsl[1])
        return outputs