import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
def to_color(self):
    """Converts this RGBA into a Color instance which excludes alpha."""
    return Color(int(self.red * Color.MAX_VALUE), int(self.green * Color.MAX_VALUE), int(self.blue * Color.MAX_VALUE))