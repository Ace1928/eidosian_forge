import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __cyrillic_to_roman(self, word):
    """
        Transliterate a Russian word into the Roman alphabet.

        A Russian word whose letters consist of the Cyrillic
        alphabet are transliterated into the Roman alphabet
        in order to ease the forthcoming stemming process.

        :param word: The word that is transliterated.
        :type word: unicode
        :return: the transliterated word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
    word = word.replace('А', 'a').replace('а', 'a').replace('Б', 'b').replace('б', 'b').replace('В', 'v').replace('в', 'v').replace('Г', 'g').replace('г', 'g').replace('Д', 'd').replace('д', 'd').replace('Е', 'e').replace('е', 'e').replace('Ё', 'e').replace('ё', 'e').replace('Ж', 'zh').replace('ж', 'zh').replace('З', 'z').replace('з', 'z').replace('И', 'i').replace('и', 'i').replace('Й', 'i`').replace('й', 'i`').replace('К', 'k').replace('к', 'k').replace('Л', 'l').replace('л', 'l').replace('М', 'm').replace('м', 'm').replace('Н', 'n').replace('н', 'n').replace('О', 'o').replace('о', 'o').replace('П', 'p').replace('п', 'p').replace('Р', 'r').replace('р', 'r').replace('С', 's').replace('с', 's').replace('Т', 't').replace('т', 't').replace('У', 'u').replace('у', 'u').replace('Ф', 'f').replace('ф', 'f').replace('Х', 'kh').replace('х', 'kh').replace('Ц', 't^s').replace('ц', 't^s').replace('Ч', 'ch').replace('ч', 'ch').replace('Ш', 'sh').replace('ш', 'sh').replace('Щ', 'shch').replace('щ', 'shch').replace('Ъ', "''").replace('ъ', "''").replace('Ы', 'y').replace('ы', 'y').replace('Ь', "'").replace('ь', "'").replace('Э', 'e`').replace('э', 'e`').replace('Ю', 'i^u').replace('ю', 'i^u').replace('Я', 'i^a').replace('я', 'i^a')
    return word