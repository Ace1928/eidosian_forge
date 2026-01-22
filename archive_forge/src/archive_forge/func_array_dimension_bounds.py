from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def array_dimension_bounds(mindim=0, maxdim=np.inf):

    def validator(trait, value):
        dim = len(value.shape)
        if dim < mindim or dim > maxdim:
            raise TraitError('Dimension mismatch for trait %s of class %s: expected an                 array of dimension comprised in interval [%s, %s] and got an array of shape %s' % (trait.name, trait.this_class, mindim, maxdim, value.shape))
        return value
    return validator