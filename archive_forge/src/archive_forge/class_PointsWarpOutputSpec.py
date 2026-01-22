import os.path as op
import re
from ... import logging
from .base import ElastixBaseInputSpec
from ..base import CommandLine, TraitedSpec, File, traits, InputMultiPath
class PointsWarpOutputSpec(TraitedSpec):
    warped_file = File(desc='input points displaced in fixed image domain')