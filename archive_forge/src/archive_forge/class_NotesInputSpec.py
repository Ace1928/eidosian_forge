import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NotesInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dNotes', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    add = Str(desc='note to add', argstr='-a "%s"')
    add_history = Str(desc='note to add to history', argstr='-h "%s"', xor=['rep_history'])
    rep_history = Str(desc='note with which to replace history', argstr='-HH "%s"', xor=['add_history'])
    delete = traits.Int(desc='delete note number num', argstr='-d %d')
    ses = traits.Bool(desc='print to stdout the expanded notes', argstr='-ses')
    out_file = File(desc='output image file name', argstr='%s')