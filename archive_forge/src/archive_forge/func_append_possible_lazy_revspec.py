from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
@classmethod
def append_possible_lazy_revspec(cls, module_name, member_name):
    """Append a possible lazily loaded DWIM revspec.

        :param module_name: Name of the module with the revspec
        :param member_name: Name of the revspec within the module
        """
    cls._possible_revspecs.append(registry._LazyObjectGetter(module_name, member_name))