import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
class StandardFormat:
    """
    Class for reading and processing standard format marker files and strings.
    """

    def __init__(self, filename=None, encoding=None):
        self._encoding = encoding
        if filename is not None:
            self.open(filename)

    def open(self, sfm_file):
        """
        Open a standard format marker file for sequential reading.

        :param sfm_file: name of the standard format marker input file
        :type sfm_file: str
        """
        if isinstance(sfm_file, PathPointer):
            self._file = sfm_file.open(self._encoding)
        else:
            self._file = codecs.open(sfm_file, 'r', self._encoding)

    def open_string(self, s):
        """
        Open a standard format marker string for sequential reading.

        :param s: string to parse as a standard format marker input file
        :type s: str
        """
        self._file = StringIO(s)

    def raw_fields(self):
        """
        Return an iterator that returns the next field in a (marker, value)
        tuple. Linebreaks and trailing white space are preserved except
        for the final newline in each field.

        :rtype: iter(tuple(str, str))
        """
        join_string = '\n'
        line_regexp = '^%s(?:\\\\(\\S+)\\s*)?(.*)$'
        first_line_pat = re.compile(line_regexp % '(?:ï»¿)?')
        line_pat = re.compile(line_regexp % '')
        file_iter = iter(self._file)
        try:
            line = next(file_iter)
        except StopIteration:
            return
        mobj = re.match(first_line_pat, line)
        mkr, line_value = mobj.groups()
        value_lines = [line_value]
        self.line_num = 0
        for line in file_iter:
            self.line_num += 1
            mobj = re.match(line_pat, line)
            line_mkr, line_value = mobj.groups()
            if line_mkr:
                yield (mkr, join_string.join(value_lines))
                mkr = line_mkr
                value_lines = [line_value]
            else:
                value_lines.append(line_value)
        self.line_num += 1
        yield (mkr, join_string.join(value_lines))

    def fields(self, strip=True, unwrap=True, encoding=None, errors='strict', unicode_fields=None):
        """
        Return an iterator that returns the next field in a ``(marker, value)``
        tuple, where ``marker`` and ``value`` are unicode strings if an ``encoding``
        was specified in the ``fields()`` method. Otherwise they are non-unicode strings.

        :param strip: strip trailing whitespace from the last line of each field
        :type strip: bool
        :param unwrap: Convert newlines in a field to spaces.
        :type unwrap: bool
        :param encoding: Name of an encoding to use. If it is specified then
            the ``fields()`` method returns unicode strings rather than non
            unicode strings.
        :type encoding: str or None
        :param errors: Error handling scheme for codec. Same as the ``decode()``
            builtin string method.
        :type errors: str
        :param unicode_fields: Set of marker names whose values are UTF-8 encoded.
            Ignored if encoding is None. If the whole file is UTF-8 encoded set
            ``encoding='utf8'`` and leave ``unicode_fields`` with its default
            value of None.
        :type unicode_fields: sequence
        :rtype: iter(tuple(str, str))
        """
        if encoding is None and unicode_fields is not None:
            raise ValueError('unicode_fields is set but not encoding.')
        unwrap_pat = re.compile('\\n+')
        for mkr, val in self.raw_fields():
            if unwrap:
                val = unwrap_pat.sub(' ', val)
            if strip:
                val = val.rstrip()
            yield (mkr, val)

    def close(self):
        """Close a previously opened standard format marker file or string."""
        self._file.close()
        try:
            del self.line_num
        except AttributeError:
            pass