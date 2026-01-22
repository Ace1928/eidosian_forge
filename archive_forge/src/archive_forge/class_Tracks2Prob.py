import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tracks2Prob(CommandLine):
    """
    Convert a tract file into a map of the fraction of tracks to enter
    each voxel - also known as a tract density image (TDI) - in MRtrix's
    image format (.mif). This can be viewed using MRview or converted to
    Nifti using MRconvert.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> tdi = mrt.Tracks2Prob()
    >>> tdi.inputs.in_file = 'dwi_CSD_tracked.tck'
    >>> tdi.inputs.colour = True
    >>> tdi.run()                                       # doctest: +SKIP
    """
    _cmd = 'tracks2prob'
    input_spec = Tracks2ProbInputSpec
    output_spec = Tracks2ProbOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['tract_image'] = self.inputs.out_filename
        if not isdefined(outputs['tract_image']):
            outputs['tract_image'] = op.abspath(self._gen_outfilename())
        else:
            outputs['tract_image'] = os.path.abspath(outputs['tract_image'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_TDI.mif'