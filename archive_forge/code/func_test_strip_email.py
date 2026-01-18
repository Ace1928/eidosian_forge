import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_strip_email():
    src = '        >> >>> def f(x):\n        >> ...   return x+1\n        >> ... \n        >> >>> zz = f(2.5)'
    cln = '>>> def f(x):\n...   return x+1\n... \n>>> zz = f(2.5)'
    assert text.strip_email_quotes(src) == cln