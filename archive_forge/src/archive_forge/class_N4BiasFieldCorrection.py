import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class N4BiasFieldCorrection(ANTSCommand, CopyHeaderInterface):
    """
    Bias field correction.

    N4 is a variant of the popular N3 (nonparameteric nonuniform normalization)
    retrospective bias correction algorithm. Based on the assumption that the
    corruption of the low frequency bias field can be modeled as a convolution of
    the intensity histogram by a Gaussian, the basic algorithmic protocol is to
    iterate between deconvolving the intensity histogram by a Gaussian, remapping
    the intensities, and then spatially smoothing this result by a B-spline modeling
    of the bias field itself. The modifications from and improvements obtained over
    the original N3 algorithm are described in [Tustison2010]_.

    .. [Tustison2010] N. Tustison et al.,
      N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging,
      29(6):1310-1320, June 2010.

    Examples
    --------

    >>> import copy
    >>> from nipype.interfaces.ants import N4BiasFieldCorrection
    >>> n4 = N4BiasFieldCorrection()
    >>> n4.inputs.dimension = 3
    >>> n4.inputs.input_image = 'structural.nii'
    >>> n4.inputs.bspline_fitting_distance = 300
    >>> n4.inputs.shrink_factor = 3
    >>> n4.inputs.n_iterations = [50,50,30,20]
    >>> n4.cmdline
    'N4BiasFieldCorrection --bspline-fitting [ 300 ]
    -d 3 --input-image structural.nii
    --convergence [ 50x50x30x20 ] --output structural_corrected.nii
    --shrink-factor 3'

    >>> n4_2 = copy.deepcopy(n4)
    >>> n4_2.inputs.convergence_threshold = 1e-6
    >>> n4_2.cmdline
    'N4BiasFieldCorrection --bspline-fitting [ 300 ]
    -d 3 --input-image structural.nii
    --convergence [ 50x50x30x20, 1e-06 ] --output structural_corrected.nii
    --shrink-factor 3'

    >>> n4_3 = copy.deepcopy(n4_2)
    >>> n4_3.inputs.bspline_order = 5
    >>> n4_3.cmdline
    'N4BiasFieldCorrection --bspline-fitting [ 300, 5 ]
    -d 3 --input-image structural.nii
    --convergence [ 50x50x30x20, 1e-06 ] --output structural_corrected.nii
    --shrink-factor 3'

    >>> n4_4 = N4BiasFieldCorrection()
    >>> n4_4.inputs.input_image = 'structural.nii'
    >>> n4_4.inputs.save_bias = True
    >>> n4_4.inputs.dimension = 3
    >>> n4_4.cmdline
    'N4BiasFieldCorrection -d 3 --input-image structural.nii
    --output [ structural_corrected.nii, structural_bias.nii ]'

    >>> n4_5 = N4BiasFieldCorrection()
    >>> n4_5.inputs.input_image = 'structural.nii'
    >>> n4_5.inputs.dimension = 3
    >>> n4_5.inputs.histogram_sharpening = (0.12, 0.02, 200)
    >>> n4_5.cmdline
    'N4BiasFieldCorrection -d 3  --histogram-sharpening [0.12,0.02,200]
    --input-image structural.nii --output structural_corrected.nii'

    """
    _cmd = 'N4BiasFieldCorrection'
    input_spec = N4BiasFieldCorrectionInputSpec
    output_spec = N4BiasFieldCorrectionOutputSpec
    _copy_header_map = {'output_image': ('input_image', False), 'bias_image': ('input_image', True)}

    def __init__(self, *args, **kwargs):
        """Instantiate the N4BiasFieldCorrection interface."""
        self._out_bias_file = None
        super(N4BiasFieldCorrection, self).__init__(*args, **kwargs)

    def _format_arg(self, name, trait_spec, value):
        if name == 'output_image' and self._out_bias_file:
            newval = '[ %s, %s ]' % (value, self._out_bias_file)
            return trait_spec.argstr % newval
        if name == 'bspline_fitting_distance':
            if isdefined(self.inputs.bspline_order):
                newval = '[ %g, %d ]' % (value, self.inputs.bspline_order)
            else:
                newval = '[ %g ]' % value
            return trait_spec.argstr % newval
        if name == 'n_iterations':
            if isdefined(self.inputs.convergence_threshold):
                newval = '[ %s, %g ]' % (self._format_xarray([str(elt) for elt in value]), self.inputs.convergence_threshold)
            else:
                newval = '[ %s ]' % self._format_xarray([str(elt) for elt in value])
            return trait_spec.argstr % newval
        return super(N4BiasFieldCorrection, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        skip = (skip or []) + ['save_bias', 'bias_image']
        self._out_bias_file = None
        if self.inputs.save_bias or isdefined(self.inputs.bias_image):
            bias_image = self.inputs.bias_image
            if not isdefined(bias_image):
                bias_image = fname_presuffix(os.path.basename(self.inputs.input_image), suffix='_bias')
            self._out_bias_file = bias_image
        return super(N4BiasFieldCorrection, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = super(N4BiasFieldCorrection, self)._list_outputs()
        if self._out_bias_file:
            outputs['bias_image'] = os.path.abspath(self._out_bias_file)
        return outputs