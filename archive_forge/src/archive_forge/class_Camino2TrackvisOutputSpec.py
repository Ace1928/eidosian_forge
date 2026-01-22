import os
from ...utils.filemanip import split_filename
from ..base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
class Camino2TrackvisOutputSpec(TraitedSpec):
    trackvis = File(exists=True, desc='The filename to which to write the .trk (trackvis) file.')