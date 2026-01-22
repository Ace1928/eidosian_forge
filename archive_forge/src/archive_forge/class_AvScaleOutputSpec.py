import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class AvScaleOutputSpec(TraitedSpec):
    rotation_translation_matrix = traits.List(traits.List(traits.Float), desc='Rotation and Translation Matrix')
    scales = traits.List(traits.Float, desc='Scales (x,y,z)')
    skews = traits.List(traits.Float, desc='Skews')
    average_scaling = traits.Float(desc='Average Scaling')
    determinant = traits.Float(desc='Determinant')
    forward_half_transform = traits.List(traits.List(traits.Float), desc='Forward Half Transform')
    backward_half_transform = traits.List(traits.List(traits.Float), desc='Backwards Half Transform')
    left_right_orientation_preserved = traits.Bool(desc='True if LR orientation preserved')
    rot_angles = traits.List(traits.Float, desc='rotation angles')
    translations = traits.List(traits.Float, desc='translations')