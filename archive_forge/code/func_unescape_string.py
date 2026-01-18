import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def unescape_string(self, s):
    out = self.acorn.six.u('')
    escaped = 0
    input_scan = InputScanner(s)
    matched = None
    while input_scan.hasNext():
        matched = input_scan.match(re.compile('([\\s]|[^\\\\]|\\\\\\\\)+'))
        if matched:
            out += matched.group(0)
        if input_scan.peek() != '\\':
            continue
        input_scan.next()
        if input_scan.peek() == 'x':
            matched = input_scan.match(re.compile('x([0-9A-Fa-f]{2})'))
        elif input_scan.peek() == 'u':
            matched = input_scan.match(re.compile('u([0-9A-Fa-f]{4})'))
            if not matched:
                matched = input_scan.match(re.compile('u\\{([0-9A-Fa-f]+)\\}'))
        else:
            out += '\\'
            if input_scan.hasNext():
                out += input_scan.next()
            continue
        if not matched:
            return s
        escaped = int(matched.group(1), 16)
        if escaped > 126 and escaped <= 255 and matched.group(0).startswith('x'):
            return s
        elif escaped >= 0 and escaped < 32:
            out += '\\' + matched.group(0)
        elif escaped > 1114111:
            out += '\\' + matched.group(0)
        elif escaped == 34 or escaped == 39 or escaped == 92:
            out += '\\' + chr(escaped)
        else:
            out += self.acorn.six.unichr(escaped)
    return out