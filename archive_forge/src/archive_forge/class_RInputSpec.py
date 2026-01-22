import os
from shutil import which
from .. import config
from .base import (
class RInputSpec(CommandLineInputSpec):
    """Basic expected inputs to R interface"""
    script = traits.Str(argstr='-e "%s"', desc='R code to run', mandatory=True, position=-1)
    rfile = traits.Bool(True, desc='Run R using R script', usedefault=True)
    script_file = File('pyscript.R', usedefault=True, desc='Name of file to write R code to')