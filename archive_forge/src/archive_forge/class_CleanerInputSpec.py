from ..base import (
import os
class CleanerInputSpec(CommandLineInputSpec):
    artifacts_list_file = File(exists=True, argstr='%s', position=1, mandatory=True, desc='Text file listing which ICs are artifacts; can be the output from classification or can be created manually')
    cleanup_motion = traits.Bool(argstr='-m', desc='cleanup motion confounds, looks for design.fsf for highpass filter cut-off', position=2)
    highpass = traits.Float(100, argstr='-m -h %f', usedefault=True, desc='cleanup motion confounds', position=2)
    aggressive = traits.Bool(argstr='-A', desc='Apply aggressive (full variance) cleanup, instead of the default less-aggressive (unique variance) cleanup.', position=3)
    confound_file = File(argstr='-x %s', desc='Include additional confound file.', position=4)
    confound_file_1 = File(argstr='-x %s', desc='Include additional confound file.', position=5)
    confound_file_2 = File(argstr='-x %s', desc='Include additional confound file.', position=6)