from pygments.token import Token, STANDARD_TYPES
from pygments.util import add_metaclass
@add_metaclass(StyleMeta)
class Style(object):
    background_color = '#ffffff'
    highlight_color = '#ffffcc'
    styles = {}