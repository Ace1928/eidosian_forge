from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class Tags(State):
    """ Allows states to be tagged.
        Attributes:
            tags (list): A list of tag strings. `State.is_<tag>` may be used
                to check if <tag> is in the list.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contains `tags`, assign them to the attribute.
        """
        self.tags = kwargs.pop('tags', [])
        super(Tags, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item.startswith('is_'):
            return item[3:] in self.tags
        return super(Tags, self).__getattribute__(item)