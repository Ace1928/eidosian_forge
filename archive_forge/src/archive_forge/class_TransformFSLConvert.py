import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TransformFSLConvert(MRTrix3Base):
    """
    Perform conversion between FSL's transformation matrix format to mrtrix3's.
    """
    _cmd = 'transformconvert'
    input_spec = TransformFSLConvertInputSpec
    output_spec = TransformFSLConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_transform'] = op.abspath(self.inputs.out_transform)
        return outputs