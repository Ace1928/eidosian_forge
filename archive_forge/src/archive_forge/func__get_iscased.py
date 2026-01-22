import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _get_iscased(flags):
    if not flags & SRE_FLAG_IGNORECASE:
        return None
    elif flags & SRE_FLAG_UNICODE:
        return _sre.unicode_iscased
    else:
        return _sre.ascii_iscased