import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class AddNoise(BaseInterface):
    """
    Corrupts with noise the input image.


    Example
    -------
    >>> from nipype.algorithms.misc import AddNoise
    >>> noise = AddNoise()
    >>> noise.inputs.in_file = 'T1.nii'
    >>> noise.inputs.in_mask = 'mask.nii'
    >>> noise.snr = 30.0
    >>> noise.run() # doctest: +SKIP

    """
    input_spec = AddNoiseInputSpec
    output_spec = AddNoiseOutputSpec

    def _run_interface(self, runtime):
        in_image = nb.load(self.inputs.in_file)
        in_data = in_image.get_fdata()
        snr = self.inputs.snr
        if isdefined(self.inputs.in_mask):
            in_mask = np.asanyarray(nb.load(self.inputs.in_mask).dataobj)
        else:
            in_mask = np.ones_like(in_data)
        result = self.gen_noise(in_data, mask=in_mask, snr_db=snr, dist=self.inputs.dist, bg_dist=self.inputs.bg_dist)
        res_im = nb.Nifti1Image(result, in_image.affine, in_image.header)
        res_im.to_filename(self._gen_output_filename())
        return runtime

    def _gen_output_filename(self):
        if not isdefined(self.inputs.out_file):
            _, base, ext = split_filename(self.inputs.in_file)
            out_file = os.path.abspath('%s_SNR%03.2f%s' % (base, self.inputs.snr, ext))
        else:
            out_file = self.inputs.out_file
        return out_file

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename()
        return outputs

    def gen_noise(self, image, mask=None, snr_db=10.0, dist='normal', bg_dist='normal'):
        """
        Generates a copy of an image with a certain amount of
        added gaussian noise (rayleigh for background in mask)
        """
        from math import sqrt
        snr = sqrt(np.power(10.0, snr_db / 10.0))
        if mask is None:
            mask = np.ones_like(image)
        else:
            mask[mask > 0] = 1
            mask[mask < 1] = 0
            if mask.ndim < image.ndim:
                mask = np.rollaxis(np.array([mask] * image.shape[3]), 0, 4)
        signal = image[mask > 0].reshape(-1)
        if dist == 'normal':
            signal = signal - signal.mean()
            sigma_n = sqrt(signal.var() / snr)
            noise = np.random.normal(size=image.shape, scale=sigma_n)
            if np.any(mask == 0) and bg_dist == 'rayleigh':
                bg_noise = np.random.rayleigh(size=image.shape, scale=sigma_n)
                noise[mask == 0] = bg_noise[mask == 0]
            im_noise = image + noise
        elif dist == 'rician':
            sigma_n = signal.mean() / snr
            n_1 = np.random.normal(size=image.shape, scale=sigma_n)
            n_2 = np.random.normal(size=image.shape, scale=sigma_n)
            stde_1 = n_1 / sqrt(2.0)
            stde_2 = n_2 / sqrt(2.0)
            im_noise = np.sqrt((image + stde_1) ** 2 + stde_2 ** 2)
        else:
            raise NotImplementedError('Only normal and rician distributions are supported')
        return im_noise