import re
from pygments.lexer import RegexLexer, include, words
from pygments.token import Keyword, Text, Comment, Name
from pygments.token import String, Number, Punctuation, Operator
class EzhilLexer(RegexLexer):
    """
    Lexer for `Ezhil, a Tamil script-based programming language <http://ezhillang.org>`_

    .. versionadded:: 2.1
    """
    name = 'Ezhil'
    aliases = ['ezhil']
    filenames = ['*.n']
    mimetypes = ['text/x-ezhil']
    flags = re.MULTILINE | re.UNICODE
    _TALETTERS = u'[a-zA-Z_]|[\u0b80-\u0bff]'
    tokens = {'root': [include('keywords'), ('#.*\\n', Comment.Single), ('[@+/*,^\\-%]|[!<>=]=?|&&?|\\|\\|?', Operator), (u'இல்', Operator.Word), (words((u'assert', u'max', u'min', u'நீளம்', u'சரம்_இடமாற்று', u'சரம்_கண்டுபிடி', u'பட்டியல்', u'பின்இணை', u'வரிசைப்படுத்து', u'எடு', u'தலைகீழ்', u'நீட்டிக்க', u'நுழைக்க', u'வை', u'கோப்பை_திற', u'கோப்பை_எழுது', u'கோப்பை_மூடு', u'pi', u'sin', u'cos', u'tan', u'sqrt', u'hypot', u'pow', u'exp', u'log', u'log10', u'exit'), suffix='\\b'), Name.Builtin), ('(True|False)\\b', Keyword.Constant), ('[^\\S\\n]+', Text), include('identifier'), include('literal'), ('[(){}\\[\\]:;.]', Punctuation)], 'keywords': [(u'பதிப்பி|தேர்ந்தெடு|தேர்வு|ஏதேனில்|ஆனால்|இல்லைஆனால்|இல்லை|ஆக|ஒவ்வொன்றாக|இல்|வரை|செய்|முடியேனில்|பின்கொடு|முடி|நிரல்பாகம்|தொடர்|நிறுத்து|நிரல்பாகம்', Keyword)], 'identifier': [(u'(?:' + _TALETTERS + u')(?:[0-9]|' + _TALETTERS + u')*', Name)], 'literal': [('".*?"', String), ('(?u)\\d+((\\.\\d*)?[eE][+-]?\\d+|\\.\\d*)', Number.Float), ('(?u)\\d+', Number.Integer)]}

    def __init__(self, **options):
        super(EzhilLexer, self).__init__(**options)
        self.encoding = options.get('encoding', 'utf-8')