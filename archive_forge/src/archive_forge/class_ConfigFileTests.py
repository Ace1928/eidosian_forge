import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class ConfigFileTests(TestCase):

    def from_file(self, text):
        return ConfigFile.from_file(BytesIO(text))

    def test_empty(self):
        ConfigFile()

    def test_eq(self):
        self.assertEqual(ConfigFile(), ConfigFile())

    def test_default_config(self):
        cf = self.from_file(b'[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = true\n')
        self.assertEqual(ConfigFile({(b'core',): {b'repositoryformatversion': b'0', b'filemode': b'true', b'bare': b'false', b'logallrefupdates': b'true'}}), cf)

    def test_from_file_empty(self):
        cf = self.from_file(b'')
        self.assertEqual(ConfigFile(), cf)

    def test_empty_line_before_section(self):
        cf = self.from_file(b'\n[section]\n')
        self.assertEqual(ConfigFile({(b'section',): {}}), cf)

    def test_comment_before_section(self):
        cf = self.from_file(b'# foo\n[section]\n')
        self.assertEqual(ConfigFile({(b'section',): {}}), cf)

    def test_comment_after_section(self):
        cf = self.from_file(b'[section] # foo\n')
        self.assertEqual(ConfigFile({(b'section',): {}}), cf)

    def test_comment_after_variable(self):
        cf = self.from_file(b'[section]\nbar= foo # a comment\n')
        self.assertEqual(ConfigFile({(b'section',): {b'bar': b'foo'}}), cf)

    def test_comment_character_within_value_string(self):
        cf = self.from_file(b'[section]\nbar= "foo#bar"\n')
        self.assertEqual(ConfigFile({(b'section',): {b'bar': b'foo#bar'}}), cf)

    def test_comment_character_within_section_string(self):
        cf = self.from_file(b'[branch "foo#bar"] # a comment\nbar= foo\n')
        self.assertEqual(ConfigFile({(b'branch', b'foo#bar'): {b'bar': b'foo'}}), cf)

    def test_closing_bracket_within_section_string(self):
        cf = self.from_file(b'[branch "foo]bar"] # a comment\nbar= foo\n')
        self.assertEqual(ConfigFile({(b'branch', b'foo]bar'): {b'bar': b'foo'}}), cf)

    def test_from_file_section(self):
        cf = self.from_file(b'[core]\nfoo = bar\n')
        self.assertEqual(b'bar', cf.get((b'core',), b'foo'))
        self.assertEqual(b'bar', cf.get((b'core', b'foo'), b'foo'))

    def test_from_file_multiple(self):
        cf = self.from_file(b'[core]\nfoo = bar\nfoo = blah\n')
        self.assertEqual([b'bar', b'blah'], list(cf.get_multivar((b'core',), b'foo')))
        self.assertEqual([], list(cf.get_multivar((b'core',), b'blah')))

    def test_from_file_utf8_bom(self):
        text = '[core]\nfoo = bär\n'.encode('utf-8-sig')
        cf = self.from_file(text)
        self.assertEqual(b'b\xc3\xa4r', cf.get((b'core',), b'foo'))

    def test_from_file_section_case_insensitive_lower(self):
        cf = self.from_file(b'[cOre]\nfOo = bar\n')
        self.assertEqual(b'bar', cf.get((b'core',), b'foo'))
        self.assertEqual(b'bar', cf.get((b'core', b'foo'), b'foo'))

    def test_from_file_section_case_insensitive_mixed(self):
        cf = self.from_file(b'[cOre]\nfOo = bar\n')
        self.assertEqual(b'bar', cf.get((b'core',), b'fOo'))
        self.assertEqual(b'bar', cf.get((b'cOre', b'fOo'), b'fOo'))

    def test_from_file_with_mixed_quoted(self):
        cf = self.from_file(b'[core]\nfoo = "bar"la\n')
        self.assertEqual(b'barla', cf.get((b'core',), b'foo'))

    def test_from_file_section_with_open_brackets(self):
        self.assertRaises(ValueError, self.from_file, b'[core\nfoo = bar\n')

    def test_from_file_value_with_open_quoted(self):
        self.assertRaises(ValueError, self.from_file, b'[core]\nfoo = "bar\n')

    def test_from_file_with_quotes(self):
        cf = self.from_file(b'[core]\nfoo = " bar"\n')
        self.assertEqual(b' bar', cf.get((b'core',), b'foo'))

    def test_from_file_with_interrupted_line(self):
        cf = self.from_file(b'[core]\nfoo = bar\\\n la\n')
        self.assertEqual(b'barla', cf.get((b'core',), b'foo'))

    def test_from_file_with_boolean_setting(self):
        cf = self.from_file(b'[core]\nfoo\n')
        self.assertEqual(b'true', cf.get((b'core',), b'foo'))

    def test_from_file_subsection(self):
        cf = self.from_file(b'[branch "foo"]\nfoo = bar\n')
        self.assertEqual(b'bar', cf.get((b'branch', b'foo'), b'foo'))

    def test_from_file_subsection_invalid(self):
        self.assertRaises(ValueError, self.from_file, b'[branch "foo]\nfoo = bar\n')

    def test_from_file_subsection_not_quoted(self):
        cf = self.from_file(b'[branch.foo]\nfoo = bar\n')
        self.assertEqual(b'bar', cf.get((b'branch', b'foo'), b'foo'))

    def test_write_preserve_multivar(self):
        cf = self.from_file(b'[core]\nfoo = bar\nfoo = blah\n')
        f = BytesIO()
        cf.write_to_file(f)
        self.assertEqual(b'[core]\n\tfoo = bar\n\tfoo = blah\n', f.getvalue())

    def test_write_to_file_empty(self):
        c = ConfigFile()
        f = BytesIO()
        c.write_to_file(f)
        self.assertEqual(b'', f.getvalue())

    def test_write_to_file_section(self):
        c = ConfigFile()
        c.set((b'core',), b'foo', b'bar')
        f = BytesIO()
        c.write_to_file(f)
        self.assertEqual(b'[core]\n\tfoo = bar\n', f.getvalue())

    def test_write_to_file_subsection(self):
        c = ConfigFile()
        c.set((b'branch', b'blie'), b'foo', b'bar')
        f = BytesIO()
        c.write_to_file(f)
        self.assertEqual(b'[branch "blie"]\n\tfoo = bar\n', f.getvalue())

    def test_same_line(self):
        cf = self.from_file(b'[branch.foo] foo = bar\n')
        self.assertEqual(b'bar', cf.get((b'branch', b'foo'), b'foo'))

    def test_quoted_newlines_windows(self):
        cf = self.from_file(b'[alias]\r\nc = \'!f() { \\\r\n printf \'[git commit -m \\"%s\\"]\\n\' \\"$*\\" && \\\r\n git commit -m \\"$*\\"; \\\r\n }; f\'\r\n')
        self.assertEqual(list(cf.sections()), [(b'alias',)])
        self.assertEqual(b'\'!f() { printf \'[git commit -m "%s"]\n\' "$*" && git commit -m "$*"', cf.get((b'alias',), b'c'))

    def test_quoted(self):
        cf = self.from_file(b'[gui]\n\tfontdiff = -family \\"Ubuntu Mono\\" -size 11 -overstrike 0\n')
        self.assertEqual(ConfigFile({(b'gui',): {b'fontdiff': b'-family "Ubuntu Mono" -size 11 -overstrike 0'}}), cf)

    def test_quoted_multiline(self):
        cf = self.from_file(b'[alias]\nwho = "!who() {\\\n  git log --no-merges --pretty=format:\'%an - %ae\' $@ | uniq -c | sort -rn;\\\n};\\\nwho"\n')
        self.assertEqual(ConfigFile({(b'alias',): {b'who': b"!who() {git log --no-merges --pretty=format:'%an - %ae' $@ | uniq -c | sort -rn;};who"}}), cf)

    def test_set_hash_gets_quoted(self):
        c = ConfigFile()
        c.set(b'xandikos', b'color', b'#665544')
        f = BytesIO()
        c.write_to_file(f)
        self.assertEqual(b'[xandikos]\n\tcolor = "#665544"\n', f.getvalue())