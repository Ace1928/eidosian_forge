import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *

        Return all words and punctuation symbols in the corpus or in the specified
        files/categories.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :param categories: a list specifying the categories whose words have
            to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        