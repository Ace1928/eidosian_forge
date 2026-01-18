import sys
from io import StringIO
import unittest
from IPython.utils.io import Tee, capture_output
def test_tee_simple():
    """Very simple check with stdout only"""
    chan = StringIO()
    text = 'Hello'
    tee = Tee(chan, channel='stdout')
    print(text, file=chan)
    assert chan.getvalue() == text + '\n'
    tee.close()