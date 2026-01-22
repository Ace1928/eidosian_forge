from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchDutch(SearchLanguage):
    lang = 'nl'
    language_name = 'Dutch'
    js_stemmer_rawcode = 'dutch-stemmer.js'
    stopwords = dutch_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('dutch')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())