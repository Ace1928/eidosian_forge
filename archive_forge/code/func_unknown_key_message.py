import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def unknown_key_message(count):
    """returns message for number of commits"""
    return ngettext('{0} commit with unknown key', '{0} commits with unknown keys', count[SIGNATURE_KEY_MISSING]).format(count[SIGNATURE_KEY_MISSING])