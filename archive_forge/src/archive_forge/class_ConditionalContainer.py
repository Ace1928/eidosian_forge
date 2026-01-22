from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
class ConditionalContainer(Container):
    """
    Wrapper around any other container that can change the visibility. The
    received `filter` determines whether the given container should be
    displayed or not.

    :param content: :class:`.Container` instance.
    :param filter: :class:`~prompt_toolkit.filters.CLIFilter` instance.
    """

    def __init__(self, content, filter):
        assert isinstance(content, Container)
        self.content = content
        self.filter = to_cli_filter(filter)

    def __repr__(self):
        return 'ConditionalContainer(%r, filter=%r)' % (self.content, self.filter)

    def reset(self):
        self.content.reset()

    def preferred_width(self, cli, max_available_width):
        if self.filter(cli):
            return self.content.preferred_width(cli, max_available_width)
        else:
            return LayoutDimension.exact(0)

    def preferred_height(self, cli, width, max_available_height):
        if self.filter(cli):
            return self.content.preferred_height(cli, width, max_available_height)
        else:
            return LayoutDimension.exact(0)

    def write_to_screen(self, cli, screen, mouse_handlers, write_position):
        if self.filter(cli):
            return self.content.write_to_screen(cli, screen, mouse_handlers, write_position)

    def walk(self, cli):
        return self.content.walk(cli)