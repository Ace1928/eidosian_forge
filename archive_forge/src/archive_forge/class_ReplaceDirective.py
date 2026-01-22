import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class ReplaceDirective(Directive):
    """Implementation of the ``py:replace`` template directive.
    
    This directive replaces the element with the result of evaluating the
    value of the ``py:replace`` attribute:
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <span py:replace="bar">Hello</span>
    ... </div>''')
    >>> print(tmpl.generate(bar='Bye'))
    <div>
      Bye
    </div>
    
    This directive is equivalent to ``py:content`` combined with ``py:strip``,
    providing a less verbose way to achieve the same effect:
    
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <span py:content="bar" py:strip="">Hello</span>
    ... </div>''')
    >>> print(tmpl.generate(bar='Bye'))
    <div>
      Bye
    </div>
    """
    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('value')
        if not value:
            raise TemplateSyntaxError('missing value for "replace" directive', template.filepath, *pos[1:])
        expr = cls._parse_expr(value, template, *pos[1:])
        return (None, [(EXPR, expr, pos)])