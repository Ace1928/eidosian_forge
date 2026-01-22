from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def _content_filter_stack_provider(self):
    """A function that returns a stack of ContentFilters.

        The function takes a path (relative to the top of the tree) and a
        file-id as parameters.

        Returns: None if content filtering is not supported by this tree.
        """
    if self.supports_content_filtering():
        return self._content_filter_stack
    else:
        return None