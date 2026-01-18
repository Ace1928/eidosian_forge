import csv
import os
import shutil
import sys
import unittest
from glob import glob
from os.path import abspath, basename, dirname, exists
from os.path import join as pjoin
from os.path import splitext
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
import nibabel as nib
from ..loadsave import load
from ..orientations import aff2axcodes, inv_ornt_aff
from ..testing import assert_data_similar, assert_dt_equal, assert_re_in
from ..tmpdirs import InTemporaryDirectory
from .nibabel_data import needs_nibabel_data
from .scriptrunner import ScriptRunner
from .test_parrec import DTI_PAR_BVALS, DTI_PAR_BVECS
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLES
from .test_parrec_data import AFF_OFF, BALLS
@script_test
@needs_nibabel_data('nitest-balls1')
def test_parrec2nii_with_data():
    LAS2LPS = inv_ornt_aff([[0, 1], [1, -1], [2, 1]], (80, 80, 10))
    with InTemporaryDirectory():
        for par in glob(pjoin(BALLS, 'PARREC', '*.PAR')):
            par_root, ext = splitext(basename(par))
            if par_root == 'NA':
                continue
            run_command(['parrec2nii', par])
            conved_img = load(par_root + '.nii')
            assert aff2axcodes(conved_img.affine) == tuple('LAS')
            assert conved_img.shape[:3] == (80, 80, 10)
            nifti_fname = pjoin(BALLS, 'NIFTI', par_root + '.nii.gz')
            if exists(nifti_fname):
                philips_img = load(nifti_fname)
                assert aff2axcodes(philips_img.affine) == tuple('LPS')
                equiv_affine = conved_img.affine.dot(LAS2LPS)
                assert_almost_equal(philips_img.affine[:3, :3], equiv_affine[:3, :3], 3)
                aff_off = equiv_affine[:3, 3] - philips_img.affine[:3, 3]
                assert_almost_equal(aff_off, AFF_OFF, 3)
                vox_sizes = vox_size(philips_img.affine)
                assert np.all(np.abs(aff_off / vox_sizes) <= 0.501)
                if par_root != 'fieldmap':
                    conved_data_lps = np.flip(conved_img.dataobj, 1)
                    assert np.allclose(conved_data_lps, philips_img.dataobj)
    with InTemporaryDirectory():
        dti_par = pjoin(BALLS, 'PARREC', 'DTI.PAR')
        run_command(['parrec2nii', dti_par])
        assert exists('DTI.nii')
        assert not exists('DTI.bvals')
        assert not exists('DTI.bvecs')
        code, stdout, stderr = run_command(['parrec2nii', dti_par], check_code=False)
        assert code == 1
        run_command(['parrec2nii', '--overwrite', '--keep-trace', '--bvs', dti_par])
        bvecs_trace = np.loadtxt('DTI.bvecs').T
        bvals_trace = np.loadtxt('DTI.bvals')
        assert_almost_equal(bvals_trace, DTI_PAR_BVALS)
        img = load('DTI.nii')
        data = img.get_fdata()
        del img
        bvecs_LPS = DTI_PAR_BVECS[:, [2, 0, 1]]
        bvecs_LAS = bvecs_LPS * [1, -1, 1]
        assert_almost_equal(np.loadtxt('DTI.bvecs'), bvecs_LAS.T)
        assert not exists('DTI.dwell_time')
        code, _, _ = run_command(['parrec2nii', '--overwrite', '--dwell-time', dti_par], check_code=False)
        assert code == 1
        run_command(['parrec2nii', '--overwrite', '--dwell-time', '--field-strength', '3', dti_par])
        exp_dwell = 26 * 9.087 / (42.576 * 3.4 * 3 * 28)
        with open('DTI.dwell_time') as fobj:
            contents = fobj.read().strip()
        assert_almost_equal(float(contents), exp_dwell)
        run_command(['parrec2nii', '--overwrite', '--bvs', dti_par])
        assert exists('DTI.bvals')
        assert exists('DTI.bvecs')
        img = load('DTI.nii')
        bvecs_notrace = np.loadtxt('DTI.bvecs').T
        bvals_notrace = np.loadtxt('DTI.bvals')
        data_notrace = img.get_fdata()
        assert data_notrace.shape[-1] == len(bvecs_notrace)
        del img
        good_mask = np.logical_or((bvecs_trace != 0).any(axis=1), bvals_trace == 0)
        assert_almost_equal(data_notrace, data[..., good_mask])
        assert_almost_equal(bvals_notrace, np.array(DTI_PAR_BVALS)[good_mask])
        assert_almost_equal(bvecs_notrace, bvecs_LAS[good_mask])
        run_command(['parrec2nii', '--overwrite', '--keep-trace', '--bvs', '--strict-sort', dti_par])
        assert_almost_equal(np.loadtxt('DTI.bvals'), np.sort(DTI_PAR_BVALS))
        img = load('DTI.nii')
        data_sorted = img.get_fdata()
        assert_almost_equal(data[..., np.argsort(DTI_PAR_BVALS, kind='stable')], data_sorted)
        del img
        run_command(['parrec2nii', '--overwrite', '--volume-info', dti_par])
        assert exists('DTI.ordering.csv')
        with open('DTI.ordering.csv') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            csv_keys = next(csvreader)
            nlines = 0
            for line in csvreader:
                nlines += 1
        assert sorted(csv_keys) == ['diffusion b value number', 'gradient orientation number']
        assert nlines == 8