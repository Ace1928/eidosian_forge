import numpy as np
from nibabel.imageclasses import spatial_axes_first
Compute volume of mask image.

    Equivalent to "fslstats /path/file.nii -V"

    Parameters
    ----------
    img : ``SpatialImage``
        All voxels of the mask should be of value 1, background should have value 0.


    Returns
    -------
    volume : float
        Volume of mask expressed in mm3.

    Examples
    --------
    >>> import numpy as np
    >>> import nibabel as nb
    >>> mask_data = np.zeros((20, 20, 20), dtype='u1')
    >>> mask_data[5:15, 5:15, 5:15] = 1
    >>> nb.imagestats.mask_volume(nb.Nifti1Image(mask_data, np.eye(4)))
    1000.0
    