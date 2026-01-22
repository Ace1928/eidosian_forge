import os
import tempfile
import time
import re
def CloseChemDrawDoc():
    """force chemdraw to save the active document

  NOTE: the extension of the filename will determine the format
   used to save the file.
  """
    d = cdApp.ActiveDocument
    d.Close()