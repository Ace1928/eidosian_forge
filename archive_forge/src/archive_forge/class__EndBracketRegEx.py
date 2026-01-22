import re
import sgmllib
class _EndBracketRegEx:

    def __init__(self):
        self.endbracket = re.compile('([^\'"<>]|"[^"]*"(?=>|/|\\s|\\w+=)|\'[^\']*\'(?=>|/|\\s|\\w+=))*(?=[<>])|.*?(?=[<>])')

    def search(self, target, index=0):
        match = self.endbracket.match(target, index)
        if match is not None:
            return EndBracketMatch(match)
        return None