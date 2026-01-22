import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def _is_character_escaped(s, charpos):
    num_bslashes = 0
    while charpos > num_bslashes and s[charpos - 1 - num_bslashes] == '\\':
        num_bslashes += 1
    return num_bslashes % 2 == 1