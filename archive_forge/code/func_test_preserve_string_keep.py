from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
@pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
def test_preserve_string_keep(self):
    inp = '\n            a: |+\n              ghi\n              jkl\n\n\n            b: x\n            '
    round_trip(inp, intermediate=dict(a='ghi\njkl\n\n\n', b='x'))