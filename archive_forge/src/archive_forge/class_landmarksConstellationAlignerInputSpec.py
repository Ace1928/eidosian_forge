import os
from ...base import (
class landmarksConstellationAlignerInputSpec(CommandLineInputSpec):
    inputLandmarksPaired = File(desc='Input landmark file (.fcsv)', exists=True, argstr='--inputLandmarksPaired %s')
    outputLandmarksPaired = traits.Either(traits.Bool, File(), hash_files=False, desc='Output landmark file (.fcsv)', argstr='--outputLandmarksPaired %s')