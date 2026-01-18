import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def write_continuation_prompt(self):
    prompt_tokens = self._continuation_prompt(self._prompt_size)
    for style, text in prompt_tokens:
        self.write(text, style, mark=Tk_.INSERT, advance=True)