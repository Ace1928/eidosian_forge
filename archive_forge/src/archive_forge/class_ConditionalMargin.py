from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
class ConditionalMargin(Margin):
    """
    Wrapper around other :class:`.Margin` classes to show/hide them.
    """

    def __init__(self, margin, filter):
        assert isinstance(margin, Margin)
        self.margin = margin
        self.filter = to_cli_filter(filter)

    def get_width(self, cli, ui_content):
        if self.filter(cli):
            return self.margin.get_width(cli, ui_content)
        else:
            return 0

    def create_margin(self, cli, window_render_info, width, height):
        if width and self.filter(cli):
            return self.margin.create_margin(cli, window_render_info, width, height)
        else:
            return []