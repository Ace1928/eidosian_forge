import keyword
import os
import sys
import token
import tokenize
from IPython.utils.coloransi import TermColors, InputTermColors,ColorScheme, ColorSchemeTable
from .colorable import Colorable
from io import StringIO
 Token handler, with syntax highlighting.