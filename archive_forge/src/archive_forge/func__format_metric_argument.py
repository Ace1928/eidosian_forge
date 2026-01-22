import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
@staticmethod
def _format_metric_argument(**kwargs):
    retval = '%s[ %s, %s, %g, %d' % (kwargs['metric'], kwargs['fixed_image'], kwargs['moving_image'], kwargs['weight'], kwargs['radius_or_bins'])
    if 'sampling_strategy' in kwargs:
        sampling_strategy = kwargs['sampling_strategy']
    elif 'sampling_percentage' in kwargs:
        sampling_strategy = Registration.DEF_SAMPLING_STRATEGY
    else:
        sampling_strategy = None
    if sampling_strategy:
        retval += ', %s' % sampling_strategy
        if 'sampling_percentage' in kwargs:
            retval += ', %g' % kwargs['sampling_percentage']
    retval += ' ]'
    return retval