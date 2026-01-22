import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load_all
class TestDocument:

    def test_single_doc_begin_end(self):
        inp = '        ---\n        - a\n        - b\n        ...\n        '
        round_trip(inp, explicit_start=True, explicit_end=True)

    def test_multi_doc_begin_end(self):
        from srsly.ruamel_yaml import dump_all, RoundTripDumper
        inp = '        ---\n        - a\n        ...\n        ---\n        - b\n        ...\n        '
        docs = list(round_trip_load_all(inp))
        assert docs == [['a'], ['b']]
        out = dump_all(docs, Dumper=RoundTripDumper, explicit_start=True, explicit_end=True)
        assert out == '---\n- a\n...\n---\n- b\n...\n'

    def test_multi_doc_no_start(self):
        inp = '        - a\n        ...\n        ---\n        - b\n        ...\n        '
        docs = list(round_trip_load_all(inp))
        assert docs == [['a'], ['b']]

    def test_multi_doc_no_end(self):
        inp = '        - a\n        ---\n        - b\n        '
        docs = list(round_trip_load_all(inp))
        assert docs == [['a'], ['b']]

    def test_multi_doc_ends_only(self):
        inp = '        - a\n        ...\n        - b\n        ...\n        '
        docs = list(round_trip_load_all(inp, version=(1, 2)))
        assert docs == [['a'], ['b']]

    def test_multi_doc_ends_only_1_1(self):
        from srsly.ruamel_yaml import parser
        with pytest.raises(parser.ParserError):
            inp = '            - a\n            ...\n            - b\n            ...\n            '
            docs = list(round_trip_load_all(inp, version=(1, 1)))
            assert docs == [['a'], ['b']]