from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchFinnish(SearchLanguage):
    lang = 'fi'
    language_name = 'Finnish'
    js_stemmer_rawcode = 'finnish-stemmer.js'
    stopwords = finnish_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('finnish')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())