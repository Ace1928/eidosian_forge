from os.path import basename, isfile, join
from unittest.mock import MagicMock, Mock, patch
import numpy
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel
from nibabel.cmdline import parrec2nii
from nibabel.tests.test_parrec import EG_PAR, VARY_PAR
from nibabel.tmpdirs import InTemporaryDirectory
@patch('nibabel.cmdline.parrec2nii.verbose')
@patch('nibabel.cmdline.parrec2nii.io_orientation')
@patch('nibabel.cmdline.parrec2nii.nifti1')
@patch('nibabel.cmdline.parrec2nii.pr')
def test_parrec2nii_sets_qform_sform_code1(*args):
    parrec2nii.verbose.switch = False
    parrec2nii.io_orientation.return_value = [[0, 1], [1, 1], [2, 1]]
    nimg = Mock()
    nhdr = MagicMock()
    nimg.header = nhdr
    parrec2nii.nifti1.Nifti1Image.return_value = nimg
    pr_img = Mock()
    pr_hdr = Mock()
    pr_hdr.get_data_scaling.return_value = (npa([]), npa([]))
    pr_hdr.get_bvals_bvecs.return_value = (None, None)
    pr_hdr.get_affine.return_value = AN_OLD_AFFINE
    pr_img.header = pr_hdr
    parrec2nii.pr.load.return_value = pr_img
    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False
    infile = 'nonexistent.PAR'
    parrec2nii.proc_file(infile, opts)
    nhdr.set_qform.assert_called_with(AN_OLD_AFFINE, code=1)
    nhdr.set_sform.assert_called_with(AN_OLD_AFFINE, code=1)