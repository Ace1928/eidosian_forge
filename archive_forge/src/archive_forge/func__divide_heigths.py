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
def _divide_heigths(self, cli, write_position):
    """
        Return the heights for all rows.
        Or None when there is not enough space.
        """
    if not self.children:
        return []
    given_dimensions = self.get_dimensions(cli) if self.get_dimensions else None

    def get_dimension_for_child(c, index):
        if given_dimensions and given_dimensions[index] is not None:
            return given_dimensions[index]
        else:
            return c.preferred_height(cli, write_position.width, write_position.extended_height)
    dimensions = [get_dimension_for_child(c, index) for index, c in enumerate(self.children)]
    sum_dimensions = sum_layout_dimensions(dimensions)
    if sum_dimensions.min > write_position.extended_height:
        return
    sizes = [d.min for d in dimensions]
    child_generator = take_using_weights(items=list(range(len(dimensions))), weights=[d.weight for d in dimensions])
    i = next(child_generator)
    while sum(sizes) < min(write_position.extended_height, sum_dimensions.preferred):
        if sizes[i] < dimensions[i].preferred:
            sizes[i] += 1
        i = next(child_generator)
    if not any([cli.is_returning, cli.is_exiting, cli.is_aborting]):
        while sum(sizes) < min(write_position.height, sum_dimensions.max):
            if sizes[i] < dimensions[i].max:
                sizes[i] += 1
            i = next(child_generator)
    return sizes