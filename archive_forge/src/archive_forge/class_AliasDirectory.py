from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class AliasDirectory(Directory):
    """Directory lookup for locations associated with a branch.

    :parent, :submit, :public, :push, :this, and :bound are currently
    supported.  On error, a subclass of DirectoryLookupFailure will be raised.
    """
    branch_aliases = registry.Registry[str, Callable[[_mod_branch.Branch], Optional[str]]]()
    branch_aliases.register('parent', lambda b: b.get_parent(), help='The parent of this branch.')
    branch_aliases.register('submit', lambda b: b.get_submit_branch(), help='The submit branch for this branch.')
    branch_aliases.register('public', lambda b: b.get_public_branch(), help='The public location of this branch.')
    branch_aliases.register('bound', lambda b: b.get_bound_location(), help='The branch this branch is bound to, for bound branches.')
    branch_aliases.register('push', lambda b: b.get_push_location(), help='The saved location used for `brz push` with no arguments.')
    branch_aliases.register('this', lambda b: b.base, help='This branch.')

    def look_up(self, name, url, purpose=None):
        branch = _mod_branch.Branch.open_containing('.')[0]
        parts = url.split('/', 1)
        if len(parts) == 2:
            name, extra = parts
        else:
            name, = parts
            extra = None
        try:
            method = self.branch_aliases.get(name[1:])
        except KeyError:
            raise InvalidLocationAlias(url)
        else:
            result = method(branch)
        if result is None:
            raise UnsetLocationAlias(url)
        if extra is not None:
            result = urlutils.join(result, extra)
        return result

    @classmethod
    def help_text(cls, topic):
        alias_lines = []
        for key in cls.branch_aliases.keys():
            help = cls.branch_aliases.get_help(key)
            alias_lines.append('  :%-10s%s\n' % (key, help))
        return 'Location aliases\n================\n\nBazaar defines several aliases for locations associated with a branch.  These\ncan be used with most commands that expect a location, such as `brz push`.\n\nThe aliases are::\n\n%s\nFor example, to push to the parent location::\n\n    brz push :parent\n' % ''.join(alias_lines)