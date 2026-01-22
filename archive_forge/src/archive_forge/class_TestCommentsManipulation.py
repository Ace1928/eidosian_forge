from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestCommentsManipulation:

    def test_seq_set_comment_on_existing_explicit_column(self):
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        ')
        data.yaml_add_eol_comment('comment 2', key=1, column=6)
        exp = '\n        - a   # comment 1\n        - b   # comment 2\n        - c\n        '
        compare(data, exp)

    def test_seq_overwrite_comment_on_existing_explicit_column(self):
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        ')
        data.yaml_add_eol_comment('comment 2', key=0, column=6)
        exp = '\n        - a   # comment 2\n        - b\n        - c\n        '
        compare(data, exp)

    def test_seq_first_comment_explicit_column(self):
        data = load('\n        - a\n        - b\n        - c\n        ')
        data.yaml_add_eol_comment('comment 1', key=1, column=6)
        exp = '\n        - a\n        - b   # comment 1\n        - c\n        '
        compare(data, exp)

    def test_seq_set_comment_on_existing_column_prev(self):
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        - d     # comment 3\n        ')
        data.yaml_add_eol_comment('comment 2', key=1)
        exp = '\n        - a   # comment 1\n        - b   # comment 2\n        - c\n        - d     # comment 3\n        '
        compare(data, exp)

    def test_seq_set_comment_on_existing_column_next(self):
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        - d     # comment 3\n        ')
        print(data._yaml_comment)
        data.yaml_add_eol_comment('comment 2', key=2)
        exp = '\n        - a   # comment 1\n        - b\n        - c     # comment 2\n        - d     # comment 3\n        '
        compare(data, exp)

    def test_seq_set_comment_on_existing_column_further_away(self):
        """
        no comment line before or after, take the latest before
        the new position
        """
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        - d\n        - e\n        - f     # comment 3\n        ')
        print(data._yaml_comment)
        data.yaml_add_eol_comment('comment 2', key=3)
        exp = '\n        - a   # comment 1\n        - b\n        - c\n        - d   # comment 2\n        - e\n        - f     # comment 3\n        '
        compare(data, exp)

    def test_seq_set_comment_on_existing_explicit_column_with_hash(self):
        data = load('\n        - a   # comment 1\n        - b\n        - c\n        ')
        data.yaml_add_eol_comment('#  comment 2', key=1, column=6)
        exp = '\n        - a   # comment 1\n        - b   #  comment 2\n        - c\n        '
        compare(data, exp)

    def test_dict_set_comment_on_existing_explicit_column(self):
        data = load('\n        a: 1   # comment 1\n        b: 2\n        c: 3\n        d: 4\n        e: 5\n        ')
        data.yaml_add_eol_comment('comment 2', key='c', column=7)
        exp = '\n        a: 1   # comment 1\n        b: 2\n        c: 3   # comment 2\n        d: 4\n        e: 5\n        '
        compare(data, exp)

    def test_dict_overwrite_comment_on_existing_explicit_column(self):
        data = load('\n        a: 1   # comment 1\n        b: 2\n        c: 3\n        d: 4\n        e: 5\n        ')
        data.yaml_add_eol_comment('comment 2', key='a', column=7)
        exp = '\n        a: 1   # comment 2\n        b: 2\n        c: 3\n        d: 4\n        e: 5\n        '
        compare(data, exp)

    def test_map_set_comment_on_existing_column_prev(self):
        data = load('\n            a: 1   # comment 1\n            b: 2\n            c: 3\n            d: 4\n            e: 5     # comment 3\n            ')
        data.yaml_add_eol_comment('comment 2', key='b')
        exp = '\n            a: 1   # comment 1\n            b: 2   # comment 2\n            c: 3\n            d: 4\n            e: 5     # comment 3\n            '
        compare(data, exp)

    def test_map_set_comment_on_existing_column_next(self):
        data = load('\n            a: 1   # comment 1\n            b: 2\n            c: 3\n            d: 4\n            e: 5     # comment 3\n            ')
        data.yaml_add_eol_comment('comment 2', key='d')
        exp = '\n            a: 1   # comment 1\n            b: 2\n            c: 3\n            d: 4     # comment 2\n            e: 5     # comment 3\n            '
        compare(data, exp)

    def test_map_set_comment_on_existing_column_further_away(self):
        """
        no comment line before or after, take the latest before
        the new position
        """
        data = load('\n            a: 1   # comment 1\n            b: 2\n            c: 3\n            d: 4\n            e: 5     # comment 3\n            ')
        data.yaml_add_eol_comment('comment 2', key='c')
        print(round_trip_dump(data))
        exp = '\n            a: 1   # comment 1\n            b: 2\n            c: 3   # comment 2\n            d: 4\n            e: 5     # comment 3\n            '
        compare(data, exp)

    def test_before_top_map_rt(self):
        data = load('\n        a: 1\n        b: 2\n        ')
        data.yaml_set_start_comment('Hello\nWorld\n')
        exp = '\n        # Hello\n        # World\n        a: 1\n        b: 2\n        '
        compare(data, exp.format(comment='#'))

    def test_before_top_map_replace(self):
        data = load('\n        # abc\n        # def\n        a: 1 # 1\n        b: 2\n        ')
        data.yaml_set_start_comment('Hello\nWorld\n')
        exp = '\n        # Hello\n        # World\n        a: 1 # 1\n        b: 2\n        '
        compare(data, exp.format(comment='#'))

    def test_before_top_map_from_scratch(self):
        from srsly.ruamel_yaml.comments import CommentedMap
        data = CommentedMap()
        data['a'] = 1
        data['b'] = 2
        data.yaml_set_start_comment('Hello\nWorld\n')
        exp = '\n            # Hello\n            # World\n            a: 1\n            b: 2\n            '
        compare(data, exp.format(comment='#'))

    def test_before_top_seq_rt(self):
        data = load('\n        - a\n        - b\n        ')
        data.yaml_set_start_comment('Hello\nWorld\n')
        print(round_trip_dump(data))
        exp = '\n        # Hello\n        # World\n        - a\n        - b\n        '
        compare(data, exp)

    def test_before_top_seq_rt_replace(self):
        s = '\n        # this\n        # that\n        - a\n        - b\n        '
        data = load(s.format(comment='#'))
        data.yaml_set_start_comment('Hello\nWorld\n')
        print(round_trip_dump(data))
        exp = '\n        # Hello\n        # World\n        - a\n        - b\n        '
        compare(data, exp.format(comment='#'))

    def test_before_top_seq_from_scratch(self):
        from srsly.ruamel_yaml.comments import CommentedSeq
        data = CommentedSeq()
        data.append('a')
        data.append('b')
        data.yaml_set_start_comment('Hello\nWorld\n')
        print(round_trip_dump(data))
        exp = '\n        # Hello\n        # World\n        - a\n        - b\n        '
        compare(data, exp.format(comment='#'))

    def test_before_nested_map_rt(self):
        data = load('\n        a: 1\n        b:\n          c: 2\n          d: 3\n        ')
        data['b'].yaml_set_start_comment('Hello\nWorld\n')
        exp = '\n        a: 1\n        b:\n        # Hello\n        # World\n          c: 2\n          d: 3\n        '
        compare(data, exp.format(comment='#'))

    def test_before_nested_map_rt_indent(self):
        data = load('\n        a: 1\n        b:\n          c: 2\n          d: 3\n        ')
        data['b'].yaml_set_start_comment('Hello\nWorld\n', indent=2)
        exp = '\n        a: 1\n        b:\n          # Hello\n          # World\n          c: 2\n          d: 3\n        '
        compare(data, exp.format(comment='#'))
        print(data['b'].ca)

    def test_before_nested_map_from_scratch(self):
        from srsly.ruamel_yaml.comments import CommentedMap
        data = CommentedMap()
        datab = CommentedMap()
        data['a'] = 1
        data['b'] = datab
        datab['c'] = 2
        datab['d'] = 3
        data['b'].yaml_set_start_comment('Hello\nWorld\n')
        exp = '\n        a: 1\n        b:\n        # Hello\n        # World\n          c: 2\n          d: 3\n        '
        compare(data, exp.format(comment='#'))

    def test_before_nested_seq_from_scratch(self):
        from srsly.ruamel_yaml.comments import CommentedMap, CommentedSeq
        data = CommentedMap()
        datab = CommentedSeq()
        data['a'] = 1
        data['b'] = datab
        datab.append('c')
        datab.append('d')
        data['b'].yaml_set_start_comment('Hello\nWorld\n', indent=2)
        exp = '\n        a: 1\n        b:\n          # Hello\n          # World\n        - c\n        - d\n        '
        compare(data, exp.format(comment='#'))

    def test_before_nested_seq_from_scratch_block_seq_indent(self):
        from srsly.ruamel_yaml.comments import CommentedMap, CommentedSeq
        data = CommentedMap()
        datab = CommentedSeq()
        data['a'] = 1
        data['b'] = datab
        datab.append('c')
        datab.append('d')
        data['b'].yaml_set_start_comment('Hello\nWorld\n', indent=2)
        exp = '\n        a: 1\n        b:\n          # Hello\n          # World\n          - c\n          - d\n        '
        compare(data, exp.format(comment='#'), indent=4, block_seq_indent=2)

    def test_map_set_comment_before_and_after_non_first_key_00(self):
        data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n                ')
        data.yaml_set_comment_before_after_key('test1', 'before test1 (top level)', after='before test2')
        data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
        exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # before test1 (top level)\n        test1:\n          # before test2\n          test2:\n            # after test2\n            test3: 3\n        '
        compare(data, exp)

    def Xtest_map_set_comment_before_and_after_non_first_key_01(self):
        data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n        ')
        data.yaml_set_comment_before_after_key('test1', 'before test1 (top level)', after='before test2\n\n')
        data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
        exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # before test1 (top level)\n        test1:\n          # before test2\n          EOL\n          test2:\n            # after test2\n            test3: 3\n        '
        compare_eol(data, exp)

    def test_map_set_comment_before_and_after_non_first_key_01(self):
        data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n        ')
        data.yaml_set_comment_before_after_key('test1', 'before test1 (top level)', after='before test2\n\n')
        data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
        exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # before test1 (top level)\n        test1:\n          # before test2\n\n          test2:\n            # after test2\n            test3: 3\n        '
        compare(data, exp)

    def Xtest_map_set_comment_before_and_after_non_first_key_02(self):
        data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n        ')
        data.yaml_set_comment_before_after_key('test1', 'xyz\n\nbefore test1 (top level)', after='\nbefore test2', after_indent=4)
        data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
        exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # xyz\n\n        # before test1 (top level)\n        test1:\n            EOL\n            # before test2\n          test2:\n            # after test2\n            test3: 3\n        '
        compare_eol(data, exp)

    def test_map_set_comment_before_and_after_non_first_key_02(self):
        data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n        ')
        data.yaml_set_comment_before_after_key('test1', 'xyz\n\nbefore test1 (top level)', after='\nbefore test2', after_indent=4)
        data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
        exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # xyz\n\n        # before test1 (top level)\n        test1:\n\n            # before test2\n          test2:\n            # after test2\n            test3: 3\n        '
        compare(data, exp)