import os
import re as regex
from ..base import (
class ThicknessPVCInputSpec(CommandLineInputSpec):
    subjectFilePrefix = traits.Str(argstr='%s', mandatory=True, desc='Absolute path and filename prefix of the subject data')