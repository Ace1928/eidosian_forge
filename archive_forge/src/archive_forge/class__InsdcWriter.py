import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class _InsdcWriter(SequenceWriter):
    """Base class for GenBank and EMBL writers (PRIVATE)."""
    MAX_WIDTH = 80
    QUALIFIER_INDENT = 21
    QUALIFIER_INDENT_STR = ' ' * QUALIFIER_INDENT
    QUALIFIER_INDENT_TMP = '     %s                '
    FTQUAL_NO_QUOTE = ('anticodon', 'citation', 'codon_start', 'compare', 'direction', 'estimated_length', 'mod_base', 'number', 'rpt_type', 'rpt_unit_range', 'tag_peptide', 'transl_except', 'transl_table')

    def _write_feature_qualifier(self, key, value=None, quote=None):
        if value is None:
            self.handle.write(f'{self.QUALIFIER_INDENT_STR}/{key}\n')
            return
        if isinstance(value, str):
            value = value.replace('"', '""')
        if quote is None:
            if isinstance(value, int) or key in self.FTQUAL_NO_QUOTE:
                quote = False
            else:
                quote = True
        if quote:
            line = f'{self.QUALIFIER_INDENT_STR}/{key}="{value}"'
        else:
            line = f'{self.QUALIFIER_INDENT_STR}/{key}={value}'
        if len(line) <= self.MAX_WIDTH:
            self.handle.write(line + '\n')
            return
        while line.lstrip():
            if len(line) <= self.MAX_WIDTH:
                self.handle.write(line + '\n')
                return
            for index in range(min(len(line) - 1, self.MAX_WIDTH), self.QUALIFIER_INDENT + 1, -1):
                if line[index] == ' ':
                    break
            if line[index] != ' ':
                index = self.MAX_WIDTH
            assert index <= self.MAX_WIDTH
            self.handle.write(line[:index] + '\n')
            line = self.QUALIFIER_INDENT_STR + line[index:].lstrip()

    def _wrap_location(self, location):
        """Split a feature location into lines (break at commas) (PRIVATE)."""
        length = self.MAX_WIDTH - self.QUALIFIER_INDENT
        if len(location) <= length:
            return location
        index = location[:length].rfind(',')
        if index == -1:
            warnings.warn(f"Couldn't split location:\n{location}", BiopythonWarning)
            return location
        return location[:index + 1] + '\n' + self.QUALIFIER_INDENT_STR + self._wrap_location(location[index + 1:])

    def _write_feature(self, feature, record_length):
        """Write a single SeqFeature object to features table (PRIVATE)."""
        assert feature.type, feature
        location = _insdc_location_string(feature.location, record_length)
        f_type = feature.type.replace(' ', '_')
        line = (self.QUALIFIER_INDENT_TMP % f_type)[:self.QUALIFIER_INDENT] + self._wrap_location(location) + '\n'
        self.handle.write(line)
        for key, values in feature.qualifiers.items():
            if isinstance(values, (list, tuple)):
                for value in values:
                    self._write_feature_qualifier(key, value)
            else:
                self._write_feature_qualifier(key, values)

    @staticmethod
    def _get_annotation_str(record, key, default='.', just_first=False):
        """Get an annotation dictionary entry (as a string) (PRIVATE).

        Some entries are lists, in which case if just_first=True the first entry
        is returned.  If just_first=False (default) this verifies there is only
        one entry before returning it.
        """
        try:
            answer = record.annotations[key]
        except KeyError:
            return default
        if isinstance(answer, list):
            if not just_first:
                assert len(answer) == 1
            return str(answer[0])
        else:
            return str(answer)

    @staticmethod
    def _split_multi_line(text, max_len):
        """Return a list of strings (PRIVATE).

        Any single words which are too long get returned as a whole line
        (e.g. URLs) without an exception or warning.
        """
        text = text.strip()
        if len(text) <= max_len:
            return [text]
        words = text.split()
        text = ''
        while words and len(text) + 1 + len(words[0]) <= max_len:
            text += ' ' + words.pop(0)
            text = text.strip()
        answer = [text]
        while words:
            text = words.pop(0)
            while words and len(text) + 1 + len(words[0]) <= max_len:
                text += ' ' + words.pop(0)
                text = text.strip()
            answer.append(text)
        assert not words
        return answer

    def _split_contig(self, record, max_len):
        """Return a list of strings, splits on commas (PRIVATE)."""
        contig = record.annotations.get('contig', '')
        if isinstance(contig, (list, tuple)):
            contig = ''.join(contig)
        contig = self.clean(contig)
        answer = []
        while contig:
            if len(contig) > max_len:
                pos = contig[:max_len - 1].rfind(',')
                if pos == -1:
                    raise ValueError('Could not break up CONTIG')
                text, contig = (contig[:pos + 1], contig[pos + 1:])
            else:
                text, contig = (contig, '')
            answer.append(text)
        return answer