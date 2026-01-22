import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _output_filenames(self, prefix, count, transform, inverse=False):
    self.low_dimensional_transform_map = {'Rigid': 'Rigid.mat', 'Affine': 'Affine.mat', 'GenericAffine': 'GenericAffine.mat', 'CompositeAffine': 'Affine.mat', 'Similarity': 'Similarity.mat', 'Translation': 'Translation.mat', 'BSpline': 'BSpline.txt', 'Initial': 'DerivedInitialMovingTranslation.mat'}
    if transform in list(self.low_dimensional_transform_map.keys()):
        suffix = self.low_dimensional_transform_map[transform]
        inverse_mode = inverse
    else:
        inverse_mode = False
        if inverse:
            suffix = 'InverseWarp.nii.gz'
        else:
            suffix = 'Warp.nii.gz'
    return ('%s%d%s' % (prefix, count, suffix), inverse_mode)