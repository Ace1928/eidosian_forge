import os
from dulwich.errors import (
from dulwich.objects import (
from dulwich.file import (
def _follow(self, name):
    import warnings
    warnings.warn('RefsContainer._follow is deprecated. Use RefsContainer.follow instead.', DeprecationWarning)
    refnames, contents = self.follow(name)
    if not refnames:
        return (None, contents)
    return (refnames[-1], contents)