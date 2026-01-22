import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def _update_keymap(self, first_keycode, count):
    """Internal function, called to refresh the keymap cache.
        """
    lastcode = first_keycode + count
    for keysym, codes in self._keymap_syms.items():
        i = 0
        while i < len(codes):
            code = codes[i][1]
            if code >= first_keycode and code < lastcode:
                del codes[i]
            else:
                i = i + 1
    keysyms = self.get_keyboard_mapping(first_keycode, count)
    self._keymap_codes[first_keycode:lastcode] = keysyms
    code = first_keycode
    for syms in keysyms:
        index = 0
        for sym in syms:
            if sym != X.NoSymbol:
                if sym in self._keymap_syms:
                    symcodes = self._keymap_syms[sym]
                    symcodes.append((index, code))
                    symcodes.sort()
                else:
                    self._keymap_syms[sym] = [(index, code)]
            index = index + 1
        code = code + 1