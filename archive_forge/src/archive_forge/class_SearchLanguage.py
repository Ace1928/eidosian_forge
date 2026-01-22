import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
class SearchLanguage:
    """
    This class is the base class for search natural language preprocessors.  If
    you want to add support for a new language, you should override the methods
    of this class.

    You should override `lang` class property too (e.g. 'en', 'fr' and so on).

    .. attribute:: stopwords

       This is a set of stop words of the target language.  Default `stopwords`
       is empty.  This word is used for building index and embedded in JS.

    .. attribute:: js_splitter_code

       Return splitter function of JavaScript version.  The function should be
       named as ``splitQuery``.  And it should take a string and return list of
       strings.

       .. versionadded:: 3.0

    .. attribute:: js_stemmer_code

       Return stemmer class of JavaScript version.  This class' name should be
       ``Stemmer`` and this class must have ``stemWord`` method.  This string is
       embedded as-is in searchtools.js.

       This class is used to preprocess search word which Sphinx HTML readers
       type, before searching index. Default implementation does nothing.
    """
    lang: Optional[str] = None
    language_name: Optional[str] = None
    stopwords: Set[str] = set()
    js_splitter_code: str = ''
    js_stemmer_rawcode: Optional[str] = None
    js_stemmer_code = '\n/**\n * Dummy stemmer for languages without stemming rules.\n */\nvar Stemmer = function() {\n  this.stemWord = function(w) {\n    return w;\n  }\n}\n'
    _word_re = re.compile('(?u)\\w+')

    def __init__(self, options: Dict) -> None:
        self.options = options
        self.init(options)

    def init(self, options: Dict) -> None:
        """
        Initialize the class with the options the user has given.
        """

    def split(self, input: str) -> List[str]:
        """
        This method splits a sentence into words.  Default splitter splits input
        at white spaces, which should be enough for most languages except CJK
        languages.
        """
        return self._word_re.findall(input)

    def stem(self, word: str) -> str:
        """
        This method implements stemming algorithm of the Python version.

        Default implementation does nothing.  You should implement this if the
        language has any stemming rules.

        This class is used to preprocess search words before registering them in
        the search index.  The stemming of the Python version and the JS version
        (given in the js_stemmer_code attribute) must be compatible.
        """
        return word

    def word_filter(self, word: str) -> bool:
        """
        Return true if the target word should be registered in the search index.
        This method is called after stemming.
        """
        return len(word) == 0 or not (len(word) < 3 and 12353 < ord(word[0]) < 12436 or (ord(word[0]) < 256 and word in self.stopwords))