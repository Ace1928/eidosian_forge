import sys
def _promptTkinter(text='', title='', default='', root=None, timeout=None):
    """Displays a message box with text input, and OK & Cancel buttons. Returns the text entered, or None if Cancel was clicked."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    text = str(text)
    return __fillablebox(text, title, default=default, mask=None, root=root, timeout=timeout)