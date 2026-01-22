from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
class SyncFromStart(SyntaxSync):
    """
    Always start the syntax highlighting from the beginning.
    """

    def get_sync_start_position(self, document, lineno):
        return (0, 0)