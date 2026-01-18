import sys
def timeoutBoxRoot():
    global boxRoot, __replyButtonText, __enterboxText
    boxRoot.destroy()
    __replyButtonText = TIMEOUT_RETURN_VALUE
    __enterboxText = TIMEOUT_RETURN_VALUE