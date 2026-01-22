from .sage_helper import _within_sage
from .pari import *
import re
def _get_acc_prec(self, other):
    if is_exact(other):
        return (self.accuracy, self._precision)
    other_accuracy = getattr(other, 'accuracy', None)
    try:
        other_precision = other.prec()
    except AttributeError:
        if isinstance(other, Gen):
            other_precision = prec_words_to_bits(other.sizeword())
        else:
            other_precision = self._default_precision
    if is_exact(self):
        return (other_accuracy, other_precision)
    else:
        if self.accuracy is None or other_accuracy is None:
            accuracy = None
        else:
            accuracy = min(self.accuracy, other_accuracy)
        return (accuracy, min(self._precision, other_precision))