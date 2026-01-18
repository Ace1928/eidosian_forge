from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme
from .theme_gray import theme_gray

    xkcd theme

    Parameters
    ----------
    base_size : int
        Base font size. All text sizes are a scaled versions of
        the base font size.
    scale : float
        The amplitude of the wiggle perpendicular to the line (in pixels)
    length : float
        The length of the wiggle along the line (in pixels).
    randomness : float
        The factor by which the length is randomly scaled. Default is 2.
    stroke_size : float
        Size of the stroke to apply to the lines and text paths.
    stroke_color : str | tuple
        Color of the strokes. Use `"none"` for no color.
    