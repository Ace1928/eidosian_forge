import os
from ...utils.filemanip import split_filename
from ..base import (
class QBallMX(StdOutCommandLine):
    """
    Generates a reconstruction matrix for Q-Ball. Used in LinRecon with
    the same scheme file to reconstruct data.

    Examples
    --------
    To create  a linear transform matrix using Spherical Harmonics (sh).

    >>> import nipype.interfaces.camino as cam
    >>> qballmx = cam.QBallMX()
    >>> qballmx.inputs.scheme_file = 'A.scheme'
    >>> qballmx.inputs.basistype = 'sh'
    >>> qballmx.inputs.order = 6
    >>> qballmx.run()            # doctest: +SKIP

    To create  a linear transform matrix using Radial Basis Functions
    (rbf). This command uses the default setting of rbf sigma = 0.2618
    (15 degrees), data smoothing sigma = 0.1309 (7.5 degrees), rbf
    pointset 246

    >>> import nipype.interfaces.camino as cam
    >>> qballmx = cam.QBallMX()
    >>> qballmx.inputs.scheme_file = 'A.scheme'
    >>> qballmx.run()              # doctest: +SKIP

    The linear transform matrix from any of these two examples can then
    be run over each voxel using LinRecon

    >>> qballcoeffs = cam.LinRecon()
    >>> qballcoeffs.inputs.in_file = 'SubjectA.Bfloat'
    >>> qballcoeffs.inputs.scheme_file = 'A.scheme'
    >>> qballcoeffs.inputs.qball_mat = 'A_qmat.Bdouble'
    >>> qballcoeffs.inputs.normalize = True
    >>> qballcoeffs.inputs.bgmask = 'brain_mask.nii'
    >>> qballcoeffs.run()             # doctest: +SKIP

    """
    _cmd = 'qballmx'
    input_spec = QBallMXInputSpec
    output_spec = QBallMXOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['qmat'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.scheme_file)
        return name + '_qmat.Bdouble'