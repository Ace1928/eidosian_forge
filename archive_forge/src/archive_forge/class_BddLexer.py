from pygments.lexer import RegexLexer, include
from pygments.token import Comment, Keyword, Name, String, Number, Text, \
class BddLexer(RegexLexer):
    """
    Lexer for BDD(Behavior-driven development), which highlights not only
    keywords, but also comments, punctuations, strings, numbers, and variables.

    .. versionadded:: 2.11
    """
    name = 'Bdd'
    aliases = ['bdd']
    filenames = ['*.feature']
    mimetypes = ['text/x-bdd']
    step_keywords = 'Given|When|Then|Add|And|Feature|Scenario Outline|Scenario|Background|Examples|But'
    tokens = {'comments': [('^\\s*#.*$', Comment)], 'miscellaneous': [('(<|>|\\[|\\]|=|\\||:|\\(|\\)|\\{|\\}|,|\\.|;|-|_|\\$)', Punctuation), ('((?<=\\<)[^\\\\>]+(?=\\>))', Name.Variable), ('"([^\\"]*)"', String), ('^@\\S+', Name.Label)], 'numbers': [('(\\d+\\.?\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?', Number)], 'root': [('\\n|\\s+', Whitespace), (step_keywords, Keyword), include('comments'), include('miscellaneous'), include('numbers'), ('\\S+', Text)]}

    def analyse_text(self, text):
        return