import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _parse_revision_str(revstr):
    """This handles a revision string -> revno.

    This always returns a list.  The list will have one element for
    each revision specifier supplied.

    >>> _parse_revision_str('234')
    [<RevisionSpec_dwim 234>]
    >>> _parse_revision_str('234..567')
    [<RevisionSpec_dwim 234>, <RevisionSpec_dwim 567>]
    >>> _parse_revision_str('..')
    [<RevisionSpec None>, <RevisionSpec None>]
    >>> _parse_revision_str('..234')
    [<RevisionSpec None>, <RevisionSpec_dwim 234>]
    >>> _parse_revision_str('234..')
    [<RevisionSpec_dwim 234>, <RevisionSpec None>]
    >>> _parse_revision_str('234..456..789') # Maybe this should be an error
    [<RevisionSpec_dwim 234>, <RevisionSpec_dwim 456>, <RevisionSpec_dwim 789>]
    >>> _parse_revision_str('234....789') #Error ?
    [<RevisionSpec_dwim 234>, <RevisionSpec None>, <RevisionSpec_dwim 789>]
    >>> _parse_revision_str('revid:test@other.com-234234')
    [<RevisionSpec_revid revid:test@other.com-234234>]
    >>> _parse_revision_str('revid:test@other.com-234234..revid:test@other.com-234235')
    [<RevisionSpec_revid revid:test@other.com-234234>, <RevisionSpec_revid revid:test@other.com-234235>]
    >>> _parse_revision_str('revid:test@other.com-234234..23')
    [<RevisionSpec_revid revid:test@other.com-234234>, <RevisionSpec_dwim 23>]
    >>> _parse_revision_str('date:2005-04-12')
    [<RevisionSpec_date date:2005-04-12>]
    >>> _parse_revision_str('date:2005-04-12 12:24:33')
    [<RevisionSpec_date date:2005-04-12 12:24:33>]
    >>> _parse_revision_str('date:2005-04-12T12:24:33')
    [<RevisionSpec_date date:2005-04-12T12:24:33>]
    >>> _parse_revision_str('date:2005-04-12,12:24:33')
    [<RevisionSpec_date date:2005-04-12,12:24:33>]
    >>> _parse_revision_str('-5..23')
    [<RevisionSpec_dwim -5>, <RevisionSpec_dwim 23>]
    >>> _parse_revision_str('-5')
    [<RevisionSpec_dwim -5>]
    >>> _parse_revision_str('123a')
    [<RevisionSpec_dwim 123a>]
    >>> _parse_revision_str('abc')
    [<RevisionSpec_dwim abc>]
    >>> _parse_revision_str('branch:../branch2')
    [<RevisionSpec_branch branch:../branch2>]
    >>> _parse_revision_str('branch:../../branch2')
    [<RevisionSpec_branch branch:../../branch2>]
    >>> _parse_revision_str('branch:../../branch2..23')
    [<RevisionSpec_branch branch:../../branch2>, <RevisionSpec_dwim 23>]
    >>> _parse_revision_str('branch:..\\\\branch2')
    [<RevisionSpec_branch branch:..\\branch2>]
    >>> _parse_revision_str('branch:..\\\\..\\\\branch2..23')
    [<RevisionSpec_branch branch:..\\..\\branch2>, <RevisionSpec_dwim 23>]
    """
    revs = []
    sep = re.compile('\\.\\.(?![\\\\/])')
    for x in sep.split(revstr):
        revs.append(revisionspec.RevisionSpec.from_string(x or None))
    return revs