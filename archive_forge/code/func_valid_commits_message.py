import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def valid_commits_message(count):
    """returns message for number of commits"""
    return gettext('{0} commits with valid signatures').format(count[SIGNATURE_VALID])