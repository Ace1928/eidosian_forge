import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ABoverlapInputSpec(AFNICommandInputSpec):
    in_file_a = File(desc='input file A', argstr='%s', position=-3, mandatory=True, exists=True, copyfile=False)
    in_file_b = File(desc='input file B', argstr='%s', position=-2, mandatory=True, exists=True, copyfile=False)
    out_file = File(desc='collect output to a file', argstr=' |& tee %s', position=-1)
    no_automask = traits.Bool(desc='consider input datasets as masks', argstr='-no_automask')
    quiet = traits.Bool(desc='be as quiet as possible (without being entirely mute)', argstr='-quiet')
    verb = traits.Bool(desc='print out some progress reports (to stderr)', argstr='-verb')