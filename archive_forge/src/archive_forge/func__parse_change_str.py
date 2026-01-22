import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _parse_change_str(revstr):
    """Parse the revision string and return a tuple with left-most
    parent of the revision.

    >>> _parse_change_str('123')
    (<RevisionSpec_before before:123>, <RevisionSpec_dwim 123>)
    >>> _parse_change_str('123..124')
    Traceback (most recent call last):
      ...
    breezy.errors.RangeInChangeOption: Option --change does not accept revision ranges
    """
    revs = _parse_revision_str(revstr)
    if len(revs) > 1:
        raise errors.RangeInChangeOption()
    return (revisionspec.RevisionSpec.from_string('before:' + revstr), revs[0])