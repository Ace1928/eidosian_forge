import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestComments:

    def test_no_end_of_file_eol(self):
        """not excluding comments caused some problems if at the end of
        the file without a newline. First error, then included \x00 """
        x = '        - europe: 10 # abc'
        round_trip(x, extra='\n')
        with pytest.raises(AssertionError):
            round_trip(x, extra='a\n')

    def test_no_comments(self):
        round_trip('\n        - europe: 10\n        - usa:\n          - ohio: 2\n          - california: 9\n        ')

    def test_round_trip_ordering(self):
        round_trip('\n        a: 1\n        b: 2\n        c: 3\n        b1: 2\n        b2: 2\n        d: 4\n        e: 5\n        f: 6\n        ')

    def test_complex(self):
        round_trip('\n        - europe: 10 # top\n        - usa:\n          - ohio: 2\n          - california: 9 # o\n        ')

    def test_dropped(self):
        s = '        # comment\n        scalar\n        ...\n        '
        round_trip(s, 'scalar\n...\n')

    def test_main_mapping_begin_end(self):
        round_trip('\n        # C start a\n        # C start b\n        abc: 1\n        ghi: 2\n        klm: 3\n        # C end a\n        # C end b\n        ')

    def test_reindent(self):
        x = '        a:\n          b:     # comment 1\n            c: 1 # comment 2\n        '
        d = round_trip_load(x)
        y = round_trip_dump(d, indent=4)
        assert y == dedent('        a:\n            b:   # comment 1\n                c: 1 # comment 2\n        ')

    def test_main_mapping_begin_end_items_post(self):
        round_trip('\n        # C start a\n        # C start b\n        abc: 1      # abc comment\n        ghi: 2\n        klm: 3      # klm comment\n        # C end a\n        # C end b\n        ')

    def test_main_sequence_begin_end(self):
        round_trip('\n        # C start a\n        # C start b\n        - abc\n        - ghi\n        - klm\n        # C end a\n        # C end b\n        ')

    def test_main_sequence_begin_end_items_post(self):
        round_trip('\n        # C start a\n        # C start b\n        - abc      # abc comment\n        - ghi\n        - klm      # klm comment\n        # C end a\n        # C end b\n        ')

    def test_main_mapping_begin_end_complex(self):
        round_trip('\n        # C start a\n        # C start b\n        abc: 1\n        ghi: 2\n        klm:\n          3a: alpha\n          3b: beta   # it is all greek to me\n        # C end a\n        # C end b\n        ')

    def test_09(self):
        s = '        hr: # 1998 hr ranking\n          - Mark McGwire\n          - Sammy Sosa\n        rbi:\n          # 1998 rbi ranking\n          - Sammy Sosa\n          - Ken Griffey\n        '
        round_trip(s, indent=4, block_seq_indent=2)

    def test_09a(self):
        round_trip('\n        hr: # 1998 hr ranking\n        - Mark McGwire\n        - Sammy Sosa\n        rbi:\n          # 1998 rbi ranking\n        - Sammy Sosa\n        - Ken Griffey\n        ')

    def test_simple_map_middle_comment(self):
        round_trip('\n        abc: 1\n        # C 3a\n        # C 3b\n        ghi: 2\n        ')

    def test_map_in_map_0(self):
        round_trip('\n        map1: # comment 1\n          # comment 2\n          map2:\n            key1: val1\n        ')

    def test_map_in_map_1(self):
        round_trip('\n        map1:\n          # comment 1\n          map2:\n            key1: val1\n        ')

    def test_application_arguments(self):
        round_trip('\n        args:\n          username: anthon\n          passwd: secret\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test\n          loop:\n            wait: 10\n        ')

    def test_substitute(self):
        x = '\n        args:\n          username: anthon          # name\n          passwd: secret            # password\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test\n          loop:\n            wait: 10\n        '
        data = round_trip_load(x)
        data['args']['passwd'] = 'deleted password'
        x = x.replace(': secret          ', ': deleted password')
        assert round_trip_dump(data) == dedent(x)

    def test_set_comment(self):
        round_trip('\n        !!set\n        # the beginning\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')

    def test_omap_comment_roundtrip(self):
        round_trip('\n        !!omap\n        - a: 1\n        - b: 2  # two\n        - c: 3  # three\n        - d: 4\n        ')

    def test_omap_comment_roundtrip_pre_comment(self):
        round_trip('\n        !!omap\n        - a: 1\n        - b: 2  # two\n        - c: 3  # three\n        # last one\n        - d: 4\n        ')

    def test_non_ascii(self):
        round_trip('\n        verbosity: 1                  # 0 is minimal output, -1 none\n        base_url: http://gopher.net\n        special_indices: [1, 5, 8]\n        also_special:\n        - a\n        - 19\n        - 32\n        asia and europe: &asia_europe\n          Turkey: Ankara\n          Russia: Moscow\n        countries:\n          Asia:\n            <<: *asia_europe\n            Japan: Tokyo # 東京\n          Europe:\n            <<: *asia_europe\n            Spain: Madrid\n            Italy: Rome\n        ')

    def test_dump_utf8(self):
        import srsly.ruamel_yaml
        x = dedent('        ab:\n        - x  # comment\n        - y  # more comment\n        ')
        data = round_trip_load(x)
        dumper = srsly.ruamel_yaml.RoundTripDumper
        for utf in [True, False]:
            y = srsly.ruamel_yaml.dump(data, default_flow_style=False, Dumper=dumper, allow_unicode=utf)
            assert y == x

    def test_dump_unicode_utf8(self):
        import srsly.ruamel_yaml
        x = dedent(u'        ab:\n        - x  # comment\n        - y  # more comment\n        ')
        data = round_trip_load(x)
        dumper = srsly.ruamel_yaml.RoundTripDumper
        for utf in [True, False]:
            y = srsly.ruamel_yaml.dump(data, default_flow_style=False, Dumper=dumper, allow_unicode=utf)
            assert y == x

    def test_mlget_00(self):
        x = '        a:\n        - b:\n          c: 42\n        - d:\n            f: 196\n          e:\n            g: 3.14\n        '
        d = round_trip_load(x)
        assert d.mlget(['a', 1, 'd', 'f'], list_ok=True) == 196
        with pytest.raises(AssertionError):
            d.mlget(['a', 1, 'd', 'f']) == 196