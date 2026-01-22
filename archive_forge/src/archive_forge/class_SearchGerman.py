from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
class SearchGerman(SearchLanguage):
    lang = 'de'
    language_name = 'German'
    js_stemmer_rawcode = 'german-stemmer.js'
    stopwords = german_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('german')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())