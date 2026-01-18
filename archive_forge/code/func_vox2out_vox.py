from itertools import product
import numpy as np
from .affines import apply_affine
def vox2out_vox(mapped_voxels, voxel_sizes=None):
    """output-aligned shape, affine for input implied by `mapped_voxels`

    The input (voxel) space, and the affine mapping to output space, are given
    in `mapped_voxels`.

    The output space is implied by the affine, we don't need to know what that
    is, we just return something with the same (implied) output space.

    Our job is to work out another voxel space where the voxel array axes and
    the output axes are aligned (top left 3 x 3 of affine is diagonal with all
    positive entries) and which contains all the voxels of the implied input
    image at their correct output space positions, once resampled into the
    output voxel space.

    Parameters
    ----------
    mapped_voxels : object or length 2 sequence
        If object, has attributes ``shape`` giving input voxel shape, and
        ``affine`` giving mapping of input voxels to output space. If length 2
        sequence, elements are (shape, affine) with same meaning as above. The
        affine is a (4, 4) array-like.
    voxel_sizes : None or sequence
        Gives the diagonal entries of `output_affine` (except the trailing 1
        for the homogeneous coordinates) (``output_affine == np.diag(voxel_sizes
        + [1])``). If None, return identity `output_affine`.

    Returns
    -------
    output_shape : sequence
        Shape of output image that has voxel axes aligned to original image
        output space axes, and encloses all the voxel data from the original
        image implied by input shape.
    output_affine : (4, 4) array
        Affine of output image that has voxel axes aligned to the output axes
        implied by input affine. Top-left 3 x 3 part of affine is diagonal with
        all positive entries.  The entries come from `voxel_sizes` if
        specified, or are all 1.  If the image is < 3D, then the missing
        dimensions will have a 1 in the matching diagonal.
    """
    try:
        in_shape, in_affine = (mapped_voxels.shape, mapped_voxels.affine)
    except AttributeError:
        in_shape, in_affine = mapped_voxels
    n_axes = len(in_shape)
    if n_axes > 3:
        raise ValueError('This function can only deal with 3D images')
    if n_axes < 3:
        in_shape += (1,) * (3 - n_axes)
    out_vox = np.ones((3,))
    if voxel_sizes is not None:
        if not len(voxel_sizes) == n_axes:
            raise ValueError('voxel sizes length should match shape')
        if not np.all(np.array(voxel_sizes) > 0):
            raise ValueError('voxel sizes should all be positive')
        out_vox[:n_axes] = voxel_sizes
    in_mn_mx = zip([0, 0, 0], np.array(in_shape) - 1)
    in_corners = list(product(*in_mn_mx))
    out_corners = apply_affine(in_affine, in_corners)
    out_mn = out_corners.min(axis=0)
    out_mx = out_corners.max(axis=0)
    out_shape = np.ceil((out_mx - out_mn) / out_vox) + 1
    out_affine = np.diag(list(out_vox) + [1])
    out_affine[:3, 3] = out_mn
    return (tuple((int(i) for i in out_shape[:n_axes])), out_affine)