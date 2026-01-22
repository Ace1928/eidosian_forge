import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProbTrackXOutputSpec(TraitedSpec):
    log = File(exists=True, desc='path/name of a text record of the command that was run')
    fdt_paths = OutputMultiPath(File(exists=True), desc='path/name of a 3D image file containing the output connectivity distribution to the seed mask')
    way_total = File(exists=True, desc='path/name of a text file containing a single number corresponding to the total number of generated tracts that have not been rejected by inclusion/exclusion mask criteria')
    targets = traits.List(File(exists=True), desc='a list with all generated seeds_to_target files')
    particle_files = traits.List(File(exists=True), desc='Files describing all of the tract samples. Generated only if verbose is set to 2')