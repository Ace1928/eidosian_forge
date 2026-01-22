from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
class TestYpkgIndent:

    def test_00(self):
        inp = '\n        name       : nano\n        version    : 2.3.2\n        release    : 1\n        homepage   : http://www.nano-editor.org\n        source     :\n          - http://www.nano-editor.org/dist/v2.3/nano-2.3.2.tar.gz : ff30924807ea289f5b60106be8\n        license    : GPL-2.0\n        summary    : GNU nano is an easy-to-use text editor\n        builddeps  :\n          - ncurses-devel\n        description: |\n            GNU nano is an easy-to-use text editor originally designed\n            as a replacement for Pico, the ncurses-based editor from the non-free mailer\n            package Pine (itself now available under the Apache License as Alpine).\n        '
        round_trip(inp, indent=4, block_seq_indent=2, top_level_colon_align=True, prefix_colon=' ')