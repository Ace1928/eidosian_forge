import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_xml(self, c, previous_token):
    if self._options.e4x and c == '<' and self.allowRegExOrXML(previous_token):
        xmlStr = ''
        match = self._patterns.xml.read_match()
        if match and (not match.group(1)):
            rootTag = match.group(2)
            rootTag = re.sub('^{\\s+', '{', re.sub('\\s+}$', '}', rootTag))
            isCurlyRoot = rootTag.startswith('{')
            depth = 0
            while bool(match):
                isEndTag = match.group(1)
                tagName = match.group(2)
                isSingletonTag = match.groups()[-1] != '' or match.group(2)[0:8] == '![CDATA['
                if not isSingletonTag and (tagName == rootTag or (isCurlyRoot and re.sub('^{\\s+', '{', re.sub('\\s+}$', '}', tagName)))):
                    if isEndTag:
                        depth -= 1
                    else:
                        depth += 1
                xmlStr += match.group(0)
                if depth <= 0:
                    break
                match = self._patterns.xml.read_match()
            if not match:
                xmlStr += self._input.match(re.compile('[\\s\\S]*')).group(0)
            xmlStr = re.sub(self.acorn.allLineBreaks, '\n', xmlStr)
            return self._create_token(TOKEN.STRING, xmlStr)
    return None