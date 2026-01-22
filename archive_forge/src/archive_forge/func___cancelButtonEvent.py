import sys
def __cancelButtonEvent(event):
    """Handle pressing Esc by clicking the Cancel button."""
    global boxRoot, __widgetTexts, __replyButtonText
    __replyButtonText = CANCEL_TEXT
    boxRoot.quit()