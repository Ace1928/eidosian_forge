from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def array_supported_kinds(kinds='biufMSUO'):

    def validator(trait, value):
        if value.dtype.kind not in kinds:
            raise TraitError('Array type not supported for trait %s of class %s: expected a                 array of kind in list %r and got an array of type %s (kind %s)' % (trait.name, trait.this_class, list(kinds), value.dtype, value.dtype.kind))
        return value
    return validator