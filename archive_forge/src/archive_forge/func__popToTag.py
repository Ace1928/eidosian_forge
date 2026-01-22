from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def _popToTag(self, name, nsprefix=None, inclusivePop=True):
    """Pops the tag stack up to and including the most recent
        instance of the given tag.

        If there are no open tags with the given name, nothing will be
        popped.

        :param name: Pop up to the most recent tag with this name.
        :param nsprefix: The namespace prefix that goes with `name`.
        :param inclusivePop: It this is false, pops the tag stack up
          to but *not* including the most recent instqance of the
          given tag.

        """
    if name == self.ROOT_TAG_NAME:
        return
    most_recently_popped = None
    stack_size = len(self.tagStack)
    for i in range(stack_size - 1, 0, -1):
        if not self.open_tag_counter.get(name):
            break
        t = self.tagStack[i]
        if name == t.name and nsprefix == t.prefix:
            if inclusivePop:
                most_recently_popped = self.popTag()
            break
        most_recently_popped = self.popTag()
    return most_recently_popped