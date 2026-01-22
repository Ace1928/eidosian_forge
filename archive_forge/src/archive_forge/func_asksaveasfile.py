import sys, platform
from urllib.request import pathname2url
def asksaveasfile(mode='w', **options):
    """
    Ask for a filename to save as, and returned the opened file.
    Modified from tkFileDialog to more intelligently handle
    default file extensions.
    """
    if sys.platform == 'darwin':
        if platform.mac_ver()[0] < '10.15.2':
            options.pop('parent', None)
        if 'defaultextension' in options and (not 'initialfile' in options):
            options['initialfile'] = 'untitled' + options['defaultextension']
    return tkFileDialog.asksaveasfile(mode=mode, **options)