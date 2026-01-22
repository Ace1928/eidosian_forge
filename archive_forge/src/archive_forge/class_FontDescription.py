from ..overrides import override
from ..module import get_introspection_module
class FontDescription(Pango.FontDescription):

    def __new__(cls, string=None):
        if string is not None:
            return Pango.font_description_from_string(string)
        else:
            return Pango.FontDescription.__new__(cls)

    def __init__(self, *args, **kwargs):
        return super(FontDescription, self).__init__()