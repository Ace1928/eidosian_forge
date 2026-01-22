from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
def _compile(node, source=None, mode='eval', filename=None, lineno=-1, xform=None):
    if not filename:
        filename = '<string>'
    if IS_PYTHON2:
        if isinstance(filename, six.text_type):
            filename = filename.encode('utf-8', 'replace')
    elif not isinstance(filename, six.text_type):
        filename = filename.decode('utf-8', 'replace')
    if lineno <= 0:
        lineno = 1
    if xform is None:
        xform = {'eval': ExpressionASTTransformer}.get(mode, TemplateASTTransformer)
    tree = xform().visit(node)
    if mode == 'eval':
        name = '<Expression %r>' % (source or '?')
    else:
        lines = source.splitlines()
        if not lines:
            extract = ''
        else:
            extract = lines[0]
        if len(lines) > 1:
            extract += ' ...'
        name = '<Suite %r>' % extract
    new_source = ASTCodeGenerator(tree).code
    code = compile(new_source, filename, mode)
    try:
        return build_code_chunk(code, filename, name, lineno)
    except RuntimeError:
        return code