from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
class WindowPlacement(object):
    """
    Python wrapper over the L{WINDOWPLACEMENT} class.
    """

    def __init__(self, wp=None):
        """
        @type  wp: L{WindowPlacement} or L{WINDOWPLACEMENT}
        @param wp: Another window placement object.
        """
        self.flags = 0
        self.showCmd = 0
        self.ptMinPosition = Point()
        self.ptMaxPosition = Point()
        self.rcNormalPosition = Rect()
        if wp:
            self.flags = wp.flags
            self.showCmd = wp.showCmd
            self.ptMinPosition = Point(wp.ptMinPosition.x, wp.ptMinPosition.y)
            self.ptMaxPosition = Point(wp.ptMaxPosition.x, wp.ptMaxPosition.y)
            self.rcNormalPosition = Rect(wp.rcNormalPosition.left, wp.rcNormalPosition.top, wp.rcNormalPosition.right, wp.rcNormalPosition.bottom)

    @property
    def _as_parameter_(self):
        """
        Compatibility with ctypes.
        Allows passing transparently a Point object to an API call.
        """
        wp = WINDOWPLACEMENT()
        wp.length = sizeof(wp)
        wp.flags = self.flags
        wp.showCmd = self.showCmd
        wp.ptMinPosition.x = self.ptMinPosition.x
        wp.ptMinPosition.y = self.ptMinPosition.y
        wp.ptMaxPosition.x = self.ptMaxPosition.x
        wp.ptMaxPosition.y = self.ptMaxPosition.y
        wp.rcNormalPosition.left = self.rcNormalPosition.left
        wp.rcNormalPosition.top = self.rcNormalPosition.top
        wp.rcNormalPosition.right = self.rcNormalPosition.right
        wp.rcNormalPosition.bottom = self.rcNormalPosition.bottom
        return wp