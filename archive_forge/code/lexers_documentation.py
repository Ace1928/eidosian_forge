from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
 Return the tokens for a given line number. 