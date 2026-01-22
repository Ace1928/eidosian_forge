from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class Reorient(SimpleInterface):
    """Conform an image to a given orientation

    Flips and reorder the image data array so that the axes match the
    directions indicated in ``orientation``.
    The default ``RAS`` orientation corresponds to the first axis being ordered
    from left to right, the second axis from posterior to anterior, and the
    third axis from inferior to superior.

    For oblique images, the original orientation is considered to be the
    closest plumb orientation.

    No resampling is performed, and thus the output image is not de-obliqued
    or registered to any other image or template.

    The effective transform is calculated from the original affine matrix to
    the reoriented affine matrix.

    Examples
    --------

    If an image is not reoriented, the original file is not modified

    .. testsetup::

        >>> def print_affine(matrix):
        ...     print(str(matrix).replace(']', ' ').replace('[', ' '))

    >>> import numpy as np
    >>> from nipype.interfaces.image import Reorient
    >>> reorient = Reorient(orientation='LPS')
    >>> reorient.inputs.in_file = 'segmentation0.nii.gz'
    >>> res = reorient.run()
    >>> res.outputs.out_file
    'segmentation0.nii.gz'

    >>> print_affine(np.loadtxt(res.outputs.transform))
    1.  0.  0.  0.
    0.  1.  0.  0.
    0.  0.  1.  0.
    0.  0.  0.  1.

    >>> reorient.inputs.orientation = 'RAS'
    >>> res = reorient.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../segmentation0_ras.nii.gz'

    >>> print_affine(np.loadtxt(res.outputs.transform))
    -1.   0.   0.  60.
     0.  -1.   0.  72.
     0.   0.   1.   0.
     0.   0.   0.   1.

    .. testcleanup::

        >>> import os
        >>> os.unlink(res.outputs.out_file)
        >>> os.unlink(res.outputs.transform)
    """
    input_spec = ReorientInputSpec
    output_spec = ReorientOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import nibabel as nb
        from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff
        fname = self.inputs.in_file
        orig_img = nb.load(fname)
        orig_ornt = nb.io_orientation(orig_img.affine)
        targ_ornt = axcodes2ornt(self.inputs.orientation)
        transform = ornt_transform(orig_ornt, targ_ornt)
        affine_xfm = inv_ornt_aff(transform, orig_img.shape)
        if LooseVersion(nb.__version__) >= LooseVersion('2.4.0'):
            reoriented = orig_img.as_reoriented(transform)
        else:
            reoriented = _as_reoriented_backport(orig_img, transform)
        if reoriented is not orig_img:
            suffix = '_' + self.inputs.orientation.lower()
            out_name = fname_presuffix(fname, suffix=suffix, newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname
        mat_name = fname_presuffix(fname, suffix='.mat', newpath=runtime.cwd, use_ext=False)
        np.savetxt(mat_name, affine_xfm, fmt='%.08f')
        self._results['out_file'] = out_name
        self._results['transform'] = mat_name
        return runtime