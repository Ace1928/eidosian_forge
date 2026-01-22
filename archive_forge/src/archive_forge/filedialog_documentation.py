import sys
import platform
import tkinter.filedialog as tkFileDialog

    Ask for a filename to save as, and returned the opened file.
    Modified from tkFileDialog to more intelligently handle
    default file extensions.
    