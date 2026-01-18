import unittest
import pytest
import nibabel as nib
from nibabel.cmdline.conform import main
from nibabel.optpkg import optional_package
from nibabel.testing import get_test_data
@needs_scipy
def test_nondefault(tmpdir):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmpdir / 'output.nii.gz'
    out_shape = (100, 100, 150)
    voxel_size = (1, 2, 4)
    orientation = 'LAS'
    args = f'{infile} {outfile} --out-shape {' '.join(map(str, out_shape))} --voxel-size {' '.join(map(str, voxel_size))} --orientation {orientation}'
    main(args.split())
    assert outfile.isfile()
    c = nib.load(outfile)
    assert c.shape == out_shape
    assert c.header.get_zooms() == voxel_size
    assert nib.orientations.aff2axcodes(c.affine) == tuple(orientation)