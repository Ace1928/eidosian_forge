import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CrmshLexer(RegexLexer):
    """
    Lexer for `crmsh <http://crmsh.github.io/>`_ configuration files
    for Pacemaker clusters.

    .. versionadded:: 2.1
    """
    name = 'Crmsh'
    aliases = ['crmsh', 'pcmk']
    filenames = ['*.crmsh', '*.pcmk']
    mimetypes = []
    elem = words(('node', 'primitive', 'group', 'clone', 'ms', 'location', 'colocation', 'order', 'fencing_topology', 'rsc_ticket', 'rsc_template', 'property', 'rsc_defaults', 'op_defaults', 'acl_target', 'acl_group', 'user', 'role', 'tag'), suffix='(?![\\w#$-])')
    sub = words(('params', 'meta', 'operations', 'op', 'rule', 'attributes', 'utilization'), suffix='(?![\\w#$-])')
    acl = words(('read', 'write', 'deny'), suffix='(?![\\w#$-])')
    bin_rel = words(('and', 'or'), suffix='(?![\\w#$-])')
    un_ops = words(('defined', 'not_defined'), suffix='(?![\\w#$-])')
    date_exp = words(('in_range', 'date', 'spec', 'in'), suffix='(?![\\w#$-])')
    acl_mod = '(?:tag|ref|reference|attribute|type|xpath)'
    bin_ops = '(?:lt|gt|lte|gte|eq|ne)'
    val_qual = '(?:string|version|number)'
    rsc_role_action = '(?:Master|Started|Slave|Stopped|start|promote|demote|stop)'
    tokens = {'root': [('^#.*\\n?', Comment), ('([\\w#$-]+)(=)("(?:""|[^"])*"|\\S+)', bygroups(Name.Attribute, Punctuation, String)), ('(node)(\\s+)([\\w#$-]+)(:)', bygroups(Keyword, Whitespace, Name, Punctuation)), ('([+-]?([0-9]+|inf)):', Number), (elem, Keyword), (sub, Keyword), (acl, Keyword), ('(?:%s:)?(%s)(?![\\w#$-])' % (val_qual, bin_ops), Operator.Word), (bin_rel, Operator.Word), (un_ops, Operator.Word), (date_exp, Operator.Word), ('#[a-z]+(?![\\w#$-])', Name.Builtin), ('(%s)(:)("(?:""|[^"])*"|\\S+)' % acl_mod, bygroups(Keyword, Punctuation, Name)), ('([\\w#$-]+)(?:(:)(%s))?(?![\\w#$-])' % rsc_role_action, bygroups(Name, Punctuation, Operator.Word)), ('(\\\\(?=\\n)|[[\\](){}/:@])', Punctuation), ('\\s+|\\n', Whitespace)]}