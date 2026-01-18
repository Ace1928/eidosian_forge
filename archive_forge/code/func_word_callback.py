import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def word_callback(lexer, match):
    word = match.group()
    if re.match('.*:$', word):
        yield (match.start(), Generic.Subheading, word)
    elif re.match('(if|unless|either|any|all|while|until|loop|repeat|foreach|forall|func|function|does|has|switch|case|reduce|compose|get|set|print|prin|equal\\?|not-equal\\?|strict-equal\\?|lesser\\?|greater\\?|lesser-or-equal\\?|greater-or-equal\\?|same\\?|not|type\\?|stats|bind|union|replace|charset|routine)$', word):
        yield (match.start(), Name.Builtin, word)
    elif re.match('(make|random|reflect|to|form|mold|absolute|add|divide|multiply|negate|power|remainder|round|subtract|even\\?|odd\\?|and~|complement|or~|xor~|append|at|back|change|clear|copy|find|head|head\\?|index\\?|insert|length\\?|next|pick|poke|remove|reverse|select|sort|skip|swap|tail|tail\\?|take|trim|create|close|delete|modify|open|open\\?|query|read|rename|update|write)$', word):
        yield (match.start(), Name.Function, word)
    elif re.match('(yes|on|no|off|true|false|tab|cr|lf|newline|escape|slash|sp|space|null|none|crlf|dot|null-byte)$', word):
        yield (match.start(), Name.Builtin.Pseudo, word)
    elif re.match('(#system-global|#include|#enum|#define|#either|#if|#import|#export|#switch|#default|#get-definition)$', word):
        yield (match.start(), Keyword.Namespace, word)
    elif re.match('(system|halt|quit|quit-return|do|load|q|recycle|call|run|ask|parse|raise-error|return|exit|break|alias|push|pop|probe|\\?\\?|spec-of|body-of|quote|forever)$', word):
        yield (match.start(), Name.Exception, word)
    elif re.match('(action\\?|block\\?|char\\?|datatype\\?|file\\?|function\\?|get-path\\?|zero\\?|get-word\\?|integer\\?|issue\\?|lit-path\\?|lit-word\\?|logic\\?|native\\?|op\\?|paren\\?|path\\?|refinement\\?|set-path\\?|set-word\\?|string\\?|unset\\?|any-struct\\?|none\\?|word\\?|any-series\\?)$', word):
        yield (match.start(), Keyword, word)
    elif re.match('(JNICALL|stdcall|cdecl|infix)$', word):
        yield (match.start(), Keyword.Namespace, word)
    elif re.match('to-.*', word):
        yield (match.start(), Keyword, word)
    elif re.match('(\\+|-\\*\\*|-|\\*\\*|//|/|\\*|and|or|xor|=\\?|===|==|=|<>|<=|>=|<<<|>>>|<<|>>|<|>%)$', word):
        yield (match.start(), Operator, word)
    elif re.match('.*\\!$', word):
        yield (match.start(), Keyword.Type, word)
    elif re.match("'.*", word):
        yield (match.start(), Name.Variable.Instance, word)
    elif re.match('#.*', word):
        yield (match.start(), Name.Label, word)
    elif re.match('%.*', word):
        yield (match.start(), Name.Decorator, word)
    elif re.match(':.*', word):
        yield (match.start(), Generic.Subheading, word)
    else:
        yield (match.start(), Name.Variable, word)