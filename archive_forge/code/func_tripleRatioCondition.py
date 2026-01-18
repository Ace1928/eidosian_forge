from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def tripleRatioCondition(key_zji, key_zki, key_zli):
    tripleRatio = self[key_zji] * self[key_zki] * self[key_zli]
    if is_zero(tripleRatio - 1):
        reason = 'Triple ratio %s * %s * %s = 1' % (key_zji, key_zki, key_zli)
        return NotPU21Representation(reason)
    return True