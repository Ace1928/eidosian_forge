from pygments.style import Style
from pygments.token import Comment, Error, Generic, Keyword, Name, Number, \
class SolarizedLightStyle(SolarizedDarkStyle):
    """
    The solarized style, light.
    """
    styles = make_style(LIGHT_COLORS)
    background_color = LIGHT_COLORS['base03']
    highlight_color = LIGHT_COLORS['base02']
    line_number_color = LIGHT_COLORS['base01']
    line_number_background_color = LIGHT_COLORS['base02']