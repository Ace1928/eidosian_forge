from .._utils.registry import alias
from ..options import get_option
from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme

    A gray background with white gridlines.

    This is the default theme

    Parameters
    ----------
    base_size : int
        Base font size. All text sizes are a scaled versions of
        the base font size.
    base_family : str
        Base font family. If `None`, use [](`plotnine.options.base_family`).
    