from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
def warning_text(name, amb):
    """ The text for ambiguity warnings """
    text = '\nAmbiguities exist in dispatched function %s\n\n' % name
    text += 'The following signatures may result in ambiguous behavior:\n'
    for pair in amb:
        text += '\t' + ', '.join(('[' + str_signature(s) + ']' for s in pair)) + '\n'
    text += '\n\nConsider making the following additions:\n\n'
    text += '\n\n'.join(['@dispatch(' + str_signature(super_signature(s)) + ')\ndef %s(...)' % name for s in amb])
    return text