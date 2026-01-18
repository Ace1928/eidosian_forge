import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def voxel_sizes(self):
    """Get i, j, k voxel sizes"""
    try:
        pix_measures = self.shared.PixelMeasuresSequence[0]
    except AttributeError:
        try:
            pix_measures = self.frames[0].PixelMeasuresSequence[0]
        except AttributeError:
            raise WrapperError('Not enough data for pixel spacing')
    pix_space = pix_measures.PixelSpacing
    try:
        zs = pix_measures.SliceThickness
    except AttributeError:
        zs = self.get('SpacingBetweenSlices')
        if zs is None:
            raise WrapperError('Not enough data for slice thickness')
    return tuple(map(float, list(pix_space) + [zs]))