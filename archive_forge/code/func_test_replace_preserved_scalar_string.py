from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_replace_preserved_scalar_string(self):
    import srsly
    s = dedent('        foo: |\n          foo\n          foo\n          bar\n          foo\n        ')
    data = round_trip_load(s, preserve_quotes=True)
    so = data['foo'].replace('foo', 'bar', 2)
    assert isinstance(so, srsly.ruamel_yaml.scalarstring.LiteralScalarString)
    assert so == dedent('\n        bar\n        bar\n        bar\n        foo\n        ')