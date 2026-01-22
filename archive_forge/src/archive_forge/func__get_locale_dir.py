import gettext as _gettext
import os
import sys
def _get_locale_dir(base):
    """Returns directory to find .mo translations file in, either local or system

    :param base: plugins can specify their own local directory
    """
    if getattr(sys, 'frozen', False):
        if base is None:
            base = os.path.dirname(sys.executable)
        return os.path.join(base, 'locale')
    else:
        if base is None:
            base = os.path.dirname(__file__)
        dirpath = os.path.realpath(os.path.join(base, 'locale'))
        if os.path.exists(dirpath):
            return dirpath
    return os.path.join(sys.prefix, 'share', 'locale')