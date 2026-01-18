import tkinter.messagebox as tkMessageBox
import tkinter.simpledialog as tkSimpleDialog
from twisted.internet import task
Remove the root Tk widget from the reactor.

    Call this before destroy()ing the root widget.
    