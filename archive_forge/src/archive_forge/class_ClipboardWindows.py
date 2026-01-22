from kivy.utils import platform
from kivy.core.clipboard import ClipboardBase
import ctypes
from ctypes import wintypes
class ClipboardWindows(ClipboardBase):

    def _copy(self, data):
        self._ensure_clipboard()
        self.put(data, self._clip_mime_type)

    def get(self, mimetype='text/plain'):
        GetClipboardData = user32.GetClipboardData
        GetClipboardData.argtypes = [wintypes.UINT]
        GetClipboardData.restype = wintypes.HANDLE
        user32.OpenClipboard(user32.GetActiveWindow())
        pcontents = GetClipboardData(CF_UNICODETEXT)
        if not pcontents:
            user32.CloseClipboard()
            return ''
        pcontents_locked = GlobalLock(pcontents)
        data = c_wchar_p(pcontents_locked).value
        GlobalUnlock(pcontents)
        user32.CloseClipboard()
        return data

    def put(self, text, mimetype='text/plain'):
        SetClipboardData = user32.SetClipboardData
        SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        SetClipboardData.restype = wintypes.HANDLE
        GlobalAlloc = kernel32.GlobalAlloc
        GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        GlobalAlloc.restype = wintypes.HGLOBAL
        user32.OpenClipboard(user32.GetActiveWindow())
        user32.EmptyClipboard()
        text_len = msvcrt.wcslen(text) + 1
        hCd = GlobalAlloc(GMEM_MOVEABLE, ctypes.sizeof(ctypes.c_wchar) * text_len)
        hCd_locked = GlobalLock(hCd)
        ctypes.memmove(c_wchar_p(hCd_locked), c_wchar_p(text), ctypes.sizeof(ctypes.c_wchar) * text_len)
        GlobalUnlock(hCd)
        SetClipboardData(CF_UNICODETEXT, hCd)
        user32.CloseClipboard()

    def get_types(self):
        return ['text/plain']