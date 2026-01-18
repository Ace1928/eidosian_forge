from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_doc_indent_document_end(self):
    yaml = YAML()
    yaml.explicit_start = True
    inp = '\n        --- |-\n          some more\n          ...\n          text\n        '
    yaml.round_trip(inp)