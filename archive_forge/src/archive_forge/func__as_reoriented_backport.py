from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
def _as_reoriented_backport(img, ornt):
    """Backport of img.as_reoriented as of nibabel 2.4.0"""
    import numpy as np
    import nibabel as nb
    from nibabel.orientations import inv_ornt_aff
    if np.array_equal(ornt, [[0, 1], [1, 1], [2, 1]]):
        return img
    t_arr = nb.apply_orientation(img.dataobj, ornt)
    new_aff = img.affine.dot(inv_ornt_aff(ornt, img.shape))
    reoriented = img.__class__(t_arr, new_aff, img.header)
    if isinstance(reoriented, nb.Nifti1Pair):
        new_dim = [None if orig_dim is None else int(ornt[orig_dim, 0]) for orig_dim in img.header.get_dim_info()]
        reoriented.header.set_dim_info(*new_dim)
    return reoriented