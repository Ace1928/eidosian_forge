from .._utils.registry import alias
from ..options import get_option
from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme
@alias
class theme_grey(theme_gray):
    pass