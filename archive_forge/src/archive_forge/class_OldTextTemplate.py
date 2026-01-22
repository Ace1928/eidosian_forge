import re
import six
from genshi.core import TEXT
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.directives import *
from genshi.template.interpolation import interpolate
class OldTextTemplate(Template):
    """Legacy implementation of the old syntax text-based templates. This class
    is provided in a transition phase for backwards compatibility. New code
    should use the `NewTextTemplate` class and the improved syntax it provides.
    
    >>> tmpl = OldTextTemplate('''Dear $name,
    ... 
    ... We have the following items for you:
    ... #for item in items
    ...  * $item
    ... #end
    ... 
    ... All the best,
    ... Foobar''')
    >>> print(tmpl.generate(name='Joe', items=[1, 2, 3]).render(encoding=None))
    Dear Joe,
    <BLANKLINE>
    We have the following items for you:
     * 1
     * 2
     * 3
    <BLANKLINE>
    All the best,
    Foobar
    """
    directives = [('def', DefDirective), ('when', WhenDirective), ('otherwise', OtherwiseDirective), ('for', ForDirective), ('if', IfDirective), ('choose', ChooseDirective), ('with', WithDirective)]
    serializer = 'text'
    _DIRECTIVE_RE = re.compile('(?:^[ \\t]*(?<!\\\\)#(end).*\\n?)|(?:^[ \\t]*(?<!\\\\)#((?:\\w+|#).*)\\n?)', re.MULTILINE)

    def _parse(self, source, encoding):
        """Parse the template from text input."""
        stream = []
        dirmap = {}
        depth = 0
        source = source.read()
        if not isinstance(source, six.text_type):
            source = source.decode(encoding or 'utf-8', 'replace')
        offset = 0
        lineno = 1
        for idx, mo in enumerate(self._DIRECTIVE_RE.finditer(source)):
            start, end = mo.span()
            if start > offset:
                text = source[offset:start]
                for kind, data, pos in interpolate(text, self.filepath, lineno, lookup=self.lookup):
                    stream.append((kind, data, pos))
                lineno += len(text.splitlines())
            text = source[start:end].lstrip()[1:]
            lineno += len(text.splitlines())
            directive = text.split(None, 1)
            if len(directive) > 1:
                command, value = directive
            else:
                command, value = (directive[0], None)
            if command == 'end':
                depth -= 1
                if depth in dirmap:
                    directive, start_offset = dirmap.pop(depth)
                    substream = stream[start_offset:]
                    stream[start_offset:] = [(SUB, ([directive], substream), (self.filepath, lineno, 0))]
            elif command == 'include':
                pos = (self.filename, lineno, 0)
                stream.append((INCLUDE, (value.strip(), None, []), pos))
            elif command != '#':
                cls = self.get_directive(command)
                if cls is None:
                    raise BadDirectiveError(command)
                directive = (0, cls, value, None, (self.filepath, lineno, 0))
                dirmap[depth] = (directive, len(stream))
                depth += 1
            offset = end
        if offset < len(source):
            text = source[offset:].replace('\\#', '#')
            for kind, data, pos in interpolate(text, self.filepath, lineno, lookup=self.lookup):
                stream.append((kind, data, pos))
        return stream