from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestLiteralScalarString:

    def test_basic_string(self):
        round_trip('\n        a: abcdefg\n        ')

    def test_quoted_integer_string(self):
        round_trip("\n        a: '12345'\n        ")

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_preserve_string(self):
        inp = '\n        a: |\n          abc\n          def\n        '
        round_trip(inp, intermediate=dict(a='abc\ndef\n'))

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_preserve_string_strip(self):
        s = '\n        a: |-\n          abc\n          def\n\n        '
        round_trip(s, intermediate=dict(a='abc\ndef'))

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_preserve_string_keep(self):
        inp = '\n            a: |+\n              ghi\n              jkl\n\n\n            b: x\n            '
        round_trip(inp, intermediate=dict(a='ghi\njkl\n\n\n', b='x'))

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_preserve_string_keep_at_end(self):
        inp = '\n            a: |+\n              ghi\n              jkl\n\n            ...\n            '
        round_trip(inp, intermediate=dict(a='ghi\njkl\n\n'))

    def test_fold_string(self):
        inp = '\n        a: >\n          abc\n          def\n\n        '
        round_trip(inp)

    def test_fold_string_strip(self):
        inp = '\n        a: >-\n          abc\n          def\n\n        '
        round_trip(inp)

    def test_fold_string_keep(self):
        with pytest.raises(AssertionError) as excinfo:
            inp = '\n            a: >+\n              abc\n              def\n\n            '
            round_trip(inp, intermediate=dict(a='abc def\n\n'))