from nipype.interfaces.base import (
import os
class HistogramMatching(SEMLikeCommandLine):
    """title: Histogram Matching

    category: Filtering

    description: Normalizes the grayscale values of a source image based on the grayscale values of a reference image.  This filter uses a histogram matching technique where the histograms of the two images are matched only at a specified number of quantile values.

    The filter was originally designed to normalize MR images of the sameMR protocol and same body part. The algorithm works best if background pixels are excluded from both the source and reference histograms.  A simple background exclusion method is to exclude all pixels whose grayscale values are smaller than the mean grayscale value. ThresholdAtMeanIntensity switches on this simple background exclusion method.

    Number of match points governs the number of quantile values to be matched.

    The filter assumes that both the source and reference are of the same type and that the input and output image type have the same number of dimension and have scalar pixel types.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/HistogramMatching

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = HistogramMatchingInputSpec
    output_spec = HistogramMatchingOutputSpec
    _cmd = 'HistogramMatching '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}