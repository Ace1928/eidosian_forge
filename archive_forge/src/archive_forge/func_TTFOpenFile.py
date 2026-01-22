from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def TTFOpenFile(fn):
    """Opens a TTF file possibly after searching TTFSearchPath
    returns (filename,file)
    """
    from reportlab.lib.utils import rl_isfile, open_for_read
    try:
        f = open_for_read(fn, 'rb')
        return (fn, f)
    except IOError:
        import os
        if not os.path.isabs(fn):
            for D in _ttf_dirs(*rl_config.TTFSearchPath):
                tfn = os.path.join(D, fn)
                if rl_isfile(tfn):
                    f = open_for_read(tfn, 'rb')
                    return (tfn, f)
        raise TTFError('Can\'t open file "%s"' % fn)