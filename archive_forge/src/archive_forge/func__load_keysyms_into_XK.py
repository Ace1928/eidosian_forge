from Xlib.X import NoSymbol
def _load_keysyms_into_XK(mod):
    """keysym definition modules need no longer call Xlib.XK._load_keysyms_into_XK().
    You should remove any calls to that function from your keysym modules."""
    pass