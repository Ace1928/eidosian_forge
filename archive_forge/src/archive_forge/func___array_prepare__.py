import os
import string
import numpy as np
from Bio.File import as_handle
def __array_prepare__(self, out_arr, context=None):
    ufunc, inputs, i = context
    alphabet = self.alphabet
    for arg in inputs:
        if isinstance(arg, Array):
            if arg.alphabet != alphabet:
                raise ValueError('alphabets are inconsistent')
    return np.ndarray.__array_prepare__(self, out_arr, context)