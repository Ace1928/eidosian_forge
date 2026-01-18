import tkFont
import Tkinter
import rdkit.sping.pid
This canvas maintains a PILCanvas as its backbuffer.  Drawing calls
        are made to the backbuffer and flush() sends the image to the screen
        using TKCanvas.
           You can also save what is displayed to a file in any of the formats
        supported by PIL