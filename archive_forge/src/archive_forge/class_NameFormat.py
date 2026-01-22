from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
class NameFormat(object):
    """
    BibTeX name format string.
    
    >>> f = NameFormat('{ff~}{vv~}{ll}{, jj}')
    >>> f.parts == [
    ...     NamePart(['', 'ff', None, '']),
    ...     NamePart(['', 'vv', None, '']),
    ...     NamePart(['', 'll', None, '']),
    ...     NamePart([', ', 'jj', None, ''])
    ... ]
    True
    >>> f = NameFormat('{{ }ff~{ }}{vv~{- Test text here -}~}{ll}{, jj}')
    >>> f.parts == [
    ...     NamePart(['{ }', 'ff', None, '~{ }']),
    ...     NamePart(['', 'vv', None, '~{- Test text here -}']),
    ...     NamePart(['', 'll', None, '']),
    ...     NamePart([u', ', 'jj', None, ''])
    ... ]
    True
    >>> f = NameFormat('abc def {f~} xyz {f}?')
    >>> f.parts == [
    ...     Text('abc def '),
    ...     NamePart(['', 'f', None, '']),
    ...     Text(' xyz '),
    ...     NamePart(['', 'f', None, '']),
    ...     Text('?'),
    ... ]
    True
    >>> f = NameFormat('{{abc}{def}ff~{xyz}{#@$}}')
    >>> f.parts == [NamePart(['{abc}{def}', 'ff', None, '~{xyz}{#@$}'])]
    True
    >>> f = NameFormat('{{abc}{def}ff{xyz}{#@${}{sdf}}}')
    >>> f.parts == [NamePart(['{abc}{def}', 'ff', 'xyz', '{#@${}{sdf}}'])]
    True
    >>> f = NameFormat('{f.~}')
    >>> f.parts == [NamePart(['', 'f', None, '.'])]
    True
    >>> f = NameFormat('{f~.}')
    >>> f.parts == [NamePart(['', 'f', None, '~.'])]
    True
    >>> f = NameFormat('{f{.}~}')
    >>> f.parts == [NamePart(['', 'f', '.', ''])]
    True

    """

    def __init__(self, format):
        self.format_string = format
        self.parts = list(NameFormatParser(format).parse())

    def format(self, name):
        person = Person(name)
        return ''.join((part.format(person) for part in self.parts))

    def to_python(self):
        """Convert BibTeX name format to Python (inexactly)."""
        parts = ',\n'.join((' ' * 8 + part.to_python() for part in self.parts))
        comment = ' ' * 4 + '"""Format names similarly to %s in BibTeX."""' % self.format_string
        body = ' ' * 4 + 'return join [\n%s,\n]' % parts
        return '\n'.join(['def format_names(person, abbr=False):', comment, body])