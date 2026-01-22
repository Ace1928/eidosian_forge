import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
class MRTrix2TrackVisInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='The input file for the tracks in MRTrix (.tck) format')
    image_file = File(exists=True, desc='The image the tracks were generated from')
    matrix_file = File(exists=True, desc='A transformation matrix to apply to the tracts after they have been generated (from FLIRT - affine transformation from image_file to registration_image_file)')
    registration_image_file = File(exists=True, desc='The final image the tracks should be registered to.')
    out_filename = File('converted.trk', genfile=True, usedefault=True, desc='The output filename for the tracks in TrackVis (.trk) format')