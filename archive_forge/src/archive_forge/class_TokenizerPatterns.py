import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
class TokenizerPatterns(BaseTokenizerPatterns):

    def __init__(self, input_scanner, acorn, options):
        BaseTokenizerPatterns.__init__(self, input_scanner)
        six = __import__('six')
        self.whitespace = self.whitespace.matching(six.u('\\u00A0\\u1680\\u180e\\u2000-\\u200a\\u202f\\u205f\\u3000\\ufeff'), six.u('\\u2028\\u2029'))
        pattern = Pattern(input_scanner)
        templatable = TemplatablePattern(input_scanner).read_options(options)
        self.identifier = templatable.starting_with(acorn.identifier).matching(acorn.identifierMatch)
        self.number = pattern.matching(number_pattern)
        self.punct = pattern.matching(punct_pattern)
        self.comment = pattern.starting_with('//').until(six.u('[\\n\\r\\u2028\\u2029]'))
        self.block_comment = pattern.starting_with('/\\*').until_after('\\*/')
        self.html_comment_start = pattern.matching('<!--')
        self.html_comment_end = pattern.matching('-->')
        self.include = pattern.starting_with('#include').until_after(acorn.lineBreak)
        self.shebang = pattern.starting_with('#!').until_after(acorn.lineBreak)
        self.xml = pattern.matching(xmlRegExp)
        self.single_quote = templatable.until(six.u("['\\\\\\n\\r\\u2028\\u2029]"))
        self.double_quote = templatable.until(six.u('["\\\\\\n\\r\\u2028\\u2029]'))
        self.template_text = templatable.until('[`\\\\$]')
        self.template_expression = templatable.until('[`}\\\\]')