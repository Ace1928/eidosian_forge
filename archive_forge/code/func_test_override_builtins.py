import builtins
import os
import subprocess
import sys
import textwrap
def test_override_builtins():
    import pylab
    ok_to_override = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', 'any', 'all', 'sum', 'divmod'}
    overridden = {key for key in {*dir(pylab)} & {*dir(builtins)} if getattr(pylab, key) != getattr(builtins, key)}
    assert overridden <= ok_to_override