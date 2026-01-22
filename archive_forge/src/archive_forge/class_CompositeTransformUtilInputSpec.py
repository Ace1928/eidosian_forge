import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class CompositeTransformUtilInputSpec(ANTSCommandInputSpec):
    process = traits.Enum('assemble', 'disassemble', argstr='--%s', position=1, usedefault=True, desc='What to do with the transform inputs (assemble or disassemble)')
    out_file = File(exists=False, argstr='%s', position=2, desc='Output file path (only used for disassembly).')
    in_file = InputMultiPath(File(exists=True), mandatory=True, argstr='%s...', position=3, desc='Input transform file(s)')
    output_prefix = Str('transform', usedefault=True, argstr='%s', position=4, desc='A prefix that is prepended to all output files (only used for assembly).')