import sys
def _alertTkinter(text='', title='', button=OK_TEXT, root=None, timeout=None):
    """Displays a simple message box with text and a single OK button. Returns the text of the button clicked on."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    text = str(text)
    retVal = _buttonbox(msg=text, title=title, choices=[str(button)], root=root, timeout=timeout)
    if retVal is None:
        return button
    else:
        return retVal