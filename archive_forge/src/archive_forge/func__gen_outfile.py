import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
def _gen_outfile(self):
    if len(self.inputs.in_file) > 1 or '*' in self.inputs.in_file[0]:
        raise AttributeError('Multiple in_files found - specify either `out_file` or `out_files`.')
    _, fn, ext = split_filename(self.inputs.in_file[0])
    self.inputs.out_file = fn + '_generated' + ext
    if os.path.exists(os.path.abspath(self.inputs.out_file)):
        raise IOError('File already found - to overwrite, use `out_file`.')
    iflogger.info('Generating `out_file`.')