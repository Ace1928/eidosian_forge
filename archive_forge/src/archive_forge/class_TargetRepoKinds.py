import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
class TargetRepoKinds:
    """An enum-like set of constants.

    They are the possible values of FetchSpecFactory.target_repo_kinds.
    """
    PREEXISTING = 'preexisting'
    STACKED = 'stacked'
    EMPTY = 'empty'