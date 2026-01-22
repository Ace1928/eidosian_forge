import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
class XMLEntityEscaper:

    def __init__(self, codepoint2name, name2codepoint):
        self.codepoint2entity = {c: str('&%s;' % n) for c, n in codepoint2name.items()}
        self.name2codepoint = name2codepoint

    def escape_entities(self, text):
        """Replace characters with their character entity references.

        Only characters corresponding to a named entity are replaced.
        """
        return str(text).translate(self.codepoint2entity)

    def __escape(self, m):
        codepoint = ord(m.group())
        try:
            return self.codepoint2entity[codepoint]
        except (KeyError, IndexError):
            return '&#x%X;' % codepoint
    __escapable = re.compile('["&<>]|[^\\x00-\\x7f]')

    def escape(self, text):
        """Replace characters with their character references.

        Replace characters by their named entity references.
        Non-ASCII characters, if they do not have a named entity reference,
        are replaced by numerical character references.

        The return value is guaranteed to be ASCII.
        """
        return self.__escapable.sub(self.__escape, str(text)).encode('ascii')
    __characterrefs = re.compile('& (?:\n                                          \\#(\\d+)\n                                          | \\#x([\\da-f]+)\n                                          | ( (?!\\d) [:\\w] [-.:\\w]+ )\n                                          ) ;', re.X | re.UNICODE)

    def __unescape(self, m):
        dval, hval, name = m.groups()
        if dval:
            codepoint = int(dval)
        elif hval:
            codepoint = int(hval, 16)
        else:
            codepoint = self.name2codepoint.get(name, 65533)
        if codepoint < 128:
            return chr(codepoint)
        return chr(codepoint)

    def unescape(self, text):
        """Unescape character references.

        All character references (both entity references and numerical
        character references) are unescaped.
        """
        return self.__characterrefs.sub(self.__unescape, text)