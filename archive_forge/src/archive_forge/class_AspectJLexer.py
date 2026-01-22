import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class AspectJLexer(JavaLexer):
    """
    For `AspectJ <http://www.eclipse.org/aspectj/>`_ source code.

    .. versionadded:: 1.6
    """
    name = 'AspectJ'
    aliases = ['aspectj']
    filenames = ['*.aj']
    mimetypes = ['text/x-aspectj']
    aj_keywords = set(('aspect', 'pointcut', 'privileged', 'call', 'execution', 'initialization', 'preinitialization', 'handler', 'get', 'set', 'staticinitialization', 'target', 'args', 'within', 'withincode', 'cflow', 'cflowbelow', 'annotation', 'before', 'after', 'around', 'proceed', 'throwing', 'returning', 'adviceexecution', 'declare', 'parents', 'warning', 'error', 'soft', 'precedence', 'thisJoinPoint', 'thisJoinPointStaticPart', 'thisEnclosingJoinPointStaticPart', 'issingleton', 'perthis', 'pertarget', 'percflow', 'percflowbelow', 'pertypewithin', 'lock', 'unlock', 'thisAspectInstance'))
    aj_inter_type = set(('parents:', 'warning:', 'error:', 'soft:', 'precedence:'))
    aj_inter_type_annotation = set(('@type', '@method', '@constructor', '@field'))

    def get_tokens_unprocessed(self, text):
        for index, token, value in JavaLexer.get_tokens_unprocessed(self, text):
            if token is Name and value in self.aj_keywords:
                yield (index, Keyword, value)
            elif token is Name.Label and value in self.aj_inter_type:
                yield (index, Keyword, value[:-1])
                yield (index, Operator, value[-1])
            elif token is Name.Decorator and value in self.aj_inter_type_annotation:
                yield (index, Keyword, value)
            else:
                yield (index, token, value)