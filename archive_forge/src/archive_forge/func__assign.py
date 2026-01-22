import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
def _assign(data, value, names=_names(ast)):
    if type(names) is tuple:
        for idx in range(len(names)):
            _assign(data, value[idx], names[idx])
    else:
        data[names] = value