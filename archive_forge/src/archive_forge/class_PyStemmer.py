import warnings
import snowballstemmer
from sphinx.deprecation import RemovedInSphinx70Warning
class PyStemmer(BaseStemmer):

    def __init__(self) -> None:
        super().__init__()
        self.stemmer = snowballstemmer.stemmer('porter')

    def stem(self, word: str) -> str:
        warnings.warn(f"{self.__class__.__name__}.stem() is deprecated, use snowballstemmer.stemmer('porter').stemWord() instead.", RemovedInSphinx70Warning, stacklevel=2)
        return self.stemmer.stemWord(word)