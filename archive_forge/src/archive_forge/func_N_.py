import gettext as _gettext
import os
import sys
def N_(msg):
    """Mark message for translation but don't translate it right away."""
    return msg