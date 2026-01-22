import warnings
from bleach import ALLOWED_ATTRIBUTES, ALLOWED_TAGS, clean
from traitlets import Any, Bool, List, Set, Unicode
from .base import Preprocessor
def _get_default_css_sanitizer():
    if _USE_BLEACH_CSS_SANITIZER:
        return CSSSanitizer(allowed_css_properties=ALLOWED_STYLES)
    return None