import os
from ....base import (
class fiberstatsInputSpec(CommandLineInputSpec):
    fiber_file = File(desc='DTI Fiber File', exists=True, argstr='--fiber_file %s')
    verbose = traits.Bool(desc='produce verbose output', argstr='--verbose ')