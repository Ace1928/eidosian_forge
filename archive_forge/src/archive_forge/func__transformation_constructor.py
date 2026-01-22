import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _transformation_constructor(self):
    model = self.inputs.transformation_model
    step_length = self.inputs.gradient_step_length
    time_step = self.inputs.number_of_time_steps
    delta_time = self.inputs.delta_time
    symmetry_type = self.inputs.symmetry_type
    retval = ['--transformation-model %s' % model]
    parameters = []
    for elem in (step_length, time_step, delta_time, symmetry_type):
        if elem is not traits.Undefined:
            parameters.append('%#.2g' % elem)
    if len(parameters) > 0:
        if len(parameters) > 1:
            parameters = ','.join(parameters)
        else:
            parameters = ''.join(parameters)
        retval.append('[%s]' % parameters)
    return ''.join(retval)