import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class TitleUnsupported(errors.BzrError):
    _fmt = 'The merge proposal %(mp)s does not support a title.'