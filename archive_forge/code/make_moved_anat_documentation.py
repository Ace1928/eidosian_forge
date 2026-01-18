import numpy as np
import nibabel as nib
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat
Make anatomical image with altered affine

* Add some rotations and translations to affine;
* Save as ``.nii`` file so SPM can read it.

See ``resample_using_spm.m`` for processing of this generated image by SPM.
