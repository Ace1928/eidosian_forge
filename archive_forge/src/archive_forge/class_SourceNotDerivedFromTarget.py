import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class SourceNotDerivedFromTarget(errors.BzrError):
    """Source branch is not derived from target branch."""
    _fmt = 'Source %(source_branch)r not derived from target %(target_branch)r.'

    def __init__(self, source_branch, target_branch):
        errors.BzrError.__init__(self, source_branch=source_branch, target_branch=target_branch)