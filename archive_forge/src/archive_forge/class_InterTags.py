import contextlib
import itertools
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from . import branch as _mod_branch
from . import errors
from .inter import InterObject
from .registry import Registry
from .revision import RevisionID
class InterTags(InterObject[Tags]):
    """Operations between sets of tags.
    """
    _optimisers = []
    'The available optimised InterTags types.'

    @classmethod
    def is_compatible(klass, source: Tags, target: Tags) -> bool:
        return True

    def merge(self, overwrite: bool=False, ignore_master: bool=False, selector: Optional[TagSelector]=None) -> Tuple[TagUpdates, Set[TagConflict]]:
        """Copy tags between repositories if necessary and possible.

        This method has common command-line behaviour about handling
        error cases.

        All new definitions are copied across, except that tags that already
        exist keep their existing definitions.

        :param to_tags: Branch to receive these tags
        :param overwrite: Overwrite conflicting tags in the target branch
        :param ignore_master: Do not modify the tags in the target's master
            branch (if any).  Default is false (so the master will be updated).
        :param selector: Callback that determines whether a tag should be
            copied. It should take a tag name and as argument and return a
            boolean.

        :returns: Tuple with tag_updates and tag_conflicts.
            tag_updates is a dictionary with new tags, None is used for
            removed tags
            tag_conflicts is a set of tags that conflicted, each of which is
            (tagname, source_target, dest_target), or None if no copying was
            done.
        """
        with contextlib.ExitStack() as stack:
            if self.source.branch == self.target.branch:
                return ({}, set())
            if not self.source.branch.supports_tags():
                return ({}, set())
            source_dict = self.source.get_tag_dict()
            if not source_dict:
                return ({}, set())
            stack.enter_context(self.target.branch.lock_write())
            if ignore_master:
                master = None
            else:
                master = self.target.branch.get_master_branch()
            if master is not None:
                stack.enter_context(master.lock_write())
            updates, conflicts = self._merge_to(self.target, source_dict, overwrite, selector=selector)
            if master is not None:
                extra_updates, extra_conflicts = self._merge_to(master.tags, source_dict, overwrite, selector=selector)
                updates.update(extra_updates)
                conflicts += extra_conflicts
        return (updates, set(conflicts))

    @classmethod
    def _merge_to(cls, to_tags, source_dict, overwrite, selector):
        dest_dict = to_tags.get_tag_dict()
        result, updates, conflicts = _reconcile_tags(source_dict, dest_dict, overwrite, selector)
        if result != dest_dict:
            to_tags._set_tag_dict(result)
        return (updates, conflicts)