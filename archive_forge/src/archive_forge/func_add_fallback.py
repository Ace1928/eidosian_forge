import gettext as _gettext
import os
import sys
def add_fallback(fallback):
    """
    Add a fallback translations object.  Typically used by plugins.

    :param fallback: gettext.GNUTranslations object
    """
    install()
    _translations.add_fallback(fallback)