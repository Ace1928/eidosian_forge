from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def _ttf_dirs(*roots):
    R = _cached_ttf_dirs.get(roots, None)
    if R is None:
        join = os.path.join
        realpath = os.path.realpath
        R = []
        aR = R.append
        for root in roots:
            for r, d, f in os.walk(root, followlinks=True):
                s = realpath(r)
                if s not in R:
                    aR(s)
                for s in d:
                    s = realpath(join(r, s))
                    if s not in R:
                        aR(s)
        _cached_ttf_dirs[roots] = R
    return R