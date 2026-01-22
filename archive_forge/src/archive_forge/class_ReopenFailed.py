import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class ReopenFailed(errors.BzrError):
    _fmt = 'Reopening the merge proposal failed: %(error)s.'