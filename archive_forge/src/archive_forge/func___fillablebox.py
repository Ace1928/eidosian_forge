import sys
def __fillablebox(msg, title='', default='', mask=None, root=None, timeout=None):
    """
    Show a box in which a user can enter some text.
    You may optionally specify some default text, which will appear in the
    enterbox when it is displayed.
    Returns the text that the user entered, or None if he cancels the operation.
    """
    global boxRoot, __enterboxText, __enterboxDefaultText
    global cancelButton, entryWidget, okButton
    if title == None:
        title == ''
    if default == None:
        default = ''
    __enterboxDefaultText = default
    __enterboxText = __enterboxDefaultText
    if root:
        root.withdraw()
        boxRoot = tk.Toplevel(master=root)
        boxRoot.withdraw()
    else:
        boxRoot = tk.Tk()
        boxRoot.withdraw()
    boxRoot.title(title)
    boxRoot.iconname('Dialog')
    boxRoot.geometry(rootWindowPosition)
    boxRoot.bind('<Escape>', __enterboxCancel)
    messageFrame = tk.Frame(master=boxRoot)
    messageFrame.pack(side=tk.TOP, fill=tk.BOTH)
    buttonsFrame = tk.Frame(master=boxRoot)
    buttonsFrame.pack(side=tk.TOP, fill=tk.BOTH)
    entryFrame = tk.Frame(master=boxRoot)
    entryFrame.pack(side=tk.TOP, fill=tk.BOTH)
    buttonsFrame = tk.Frame(master=boxRoot)
    buttonsFrame.pack(side=tk.TOP, fill=tk.BOTH)
    messageWidget = tk.Message(messageFrame, width='4.5i', text=msg)
    messageWidget.configure(font=(PROPORTIONAL_FONT_FAMILY, PROPORTIONAL_FONT_SIZE))
    messageWidget.pack(side=tk.RIGHT, expand=1, fill=tk.BOTH, padx='3m', pady='3m')
    entryWidget = tk.Entry(entryFrame, width=40)
    _bindArrows(entryWidget, skipArrowKeys=True)
    entryWidget.configure(font=(PROPORTIONAL_FONT_FAMILY, TEXT_ENTRY_FONT_SIZE))
    if mask:
        entryWidget.configure(show=mask)
    entryWidget.pack(side=tk.LEFT, padx='3m')
    entryWidget.bind('<Return>', __enterboxGetText)
    entryWidget.bind('<Escape>', __enterboxCancel)
    if __enterboxDefaultText != '':
        entryWidget.insert(0, __enterboxDefaultText)
        entryWidget.select_range(0, tk.END)
    okButton = tk.Button(buttonsFrame, takefocus=1, text=OK_TEXT)
    _bindArrows(okButton)
    okButton.pack(expand=1, side=tk.LEFT, padx='3m', pady='3m', ipadx='2m', ipady='1m')
    commandButton = okButton
    handler = __enterboxGetText
    for selectionEvent in STANDARD_SELECTION_EVENTS:
        commandButton.bind('<%s>' % selectionEvent, handler)
    cancelButton = tk.Button(buttonsFrame, takefocus=1, text=CANCEL_TEXT)
    _bindArrows(cancelButton)
    cancelButton.pack(expand=1, side=tk.RIGHT, padx='3m', pady='3m', ipadx='2m', ipady='1m')
    commandButton = cancelButton
    handler = __enterboxCancel
    for selectionEvent in STANDARD_SELECTION_EVENTS:
        commandButton.bind('<%s>' % selectionEvent, handler)
    entryWidget.focus_force()
    boxRoot.deiconify()
    if timeout is not None:
        boxRoot.after(timeout, timeoutBoxRoot)
    boxRoot.mainloop()
    if root:
        root.deiconify()
    try:
        boxRoot.destroy()
    except tk.TclError:
        if __enterboxText != TIMEOUT_RETURN_VALUE:
            return None
    return __enterboxText