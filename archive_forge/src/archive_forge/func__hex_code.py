import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _hex_code(code):
    return '[%s]' % ', '.join(('%#0*x' % (_sre.CODESIZE * 2 + 2, x) for x in code))