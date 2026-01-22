from typing import Dict
import snowballstemmer
from sphinx.search import SearchLanguage
class SearchEnglish(SearchLanguage):
    lang = 'en'
    language_name = 'English'
    js_stemmer_code = js_porter_stemmer
    stopwords = english_stopwords

    def init(self, options: Dict) -> None:
        self.stemmer = snowballstemmer.stemmer('porter')

    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word.lower())