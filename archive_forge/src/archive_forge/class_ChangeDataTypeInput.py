import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class ChangeDataTypeInput(MathsInput):
    _dtypes = ['float', 'char', 'int', 'short', 'double', 'input']
    output_datatype = traits.Enum(*_dtypes, position=-1, argstr='-odt %s', mandatory=True, desc='output data type')