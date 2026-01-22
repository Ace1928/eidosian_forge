import sys, platform
from urllib.request import pathname2url
def askopenfile(parent=None):
    if sys.platform == 'darwin' and platform.mac_ver()[0] < '10.15.2':
        parent = None
    return tkFileDialog.askopenfile(parent=parent, mode='r', title='Open SnapPea Projection File', defaultextension='.lnk', filetypes=[('Link and text files', '*.lnk *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])