from ..base import (
import os
class TrainingInputSpec(CommandLineInputSpec):
    mel_icas = InputMultiPath(Directory(exists=True), copyfile=False, desc='Melodic output directories', argstr='%s', position=-1)
    trained_wts_filestem = traits.Str(desc='trained-weights filestem, used for trained_wts_file and output directories', argstr='%s', position=1)
    loo = traits.Bool(argstr='-l', desc='full leave-one-out test with classifier training', position=2)