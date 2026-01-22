import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
class AFNIPythonCommandInputSpec(CommandLineInputSpec):
    outputtype = traits.Enum('AFNI', list(Info.ftypes.keys()), desc='AFNI output filetype')
    py27_path = traits.Either('python2', File(exists=True), usedefault=True, default='python2')