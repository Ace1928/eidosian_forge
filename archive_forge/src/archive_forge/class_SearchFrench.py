from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchFrench(SearchLanguage):
    lang = 'fr'
    language_name = 'French'
    js_stemmer_rawcode = 'french-stemmer.js'
    stopwords = french_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('french')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())