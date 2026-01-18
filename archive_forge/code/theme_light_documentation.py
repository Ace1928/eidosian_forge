from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme
from .theme_gray import theme_gray

    A theme similar to [](`~plotnine.themes.theme_linedraw.theme_linedraw`)

    Has light grey lines lines and axes to direct more attention
    towards the data.

    Parameters
    ----------
    base_size : int
        Base font size. All text sizes are a scaled versions of
        the base font size.
    base_family : str
        Base font family. If `None`, use [](`plotnine.options.base_family`).
    