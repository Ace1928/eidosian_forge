from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class SVAdjustVoxSpTask(DTITKRenameMixin, SVAdjustVoxSp):
    pass