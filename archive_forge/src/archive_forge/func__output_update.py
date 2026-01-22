import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
def _output_update(self):
    """
        Update the internal property with the provided input.

        i think? updates class private attribute based on instance input
        in fsl also updates ENVIRON variable....not valid in afni
        as it uses no environment variables
        """
    self._outputtype = self.inputs.outputtype