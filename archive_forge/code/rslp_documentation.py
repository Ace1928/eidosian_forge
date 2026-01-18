from nltk.data import load
from nltk.stem.api import StemmerI

    A stemmer for Portuguese.

        >>> from nltk.stem import RSLPStemmer
        >>> st = RSLPStemmer()
        >>> # opening lines of Erico Verissimo's "Música ao Longe"
        >>> text = '''
        ... Clarissa risca com giz no quadro-negro a paisagem que os alunos
        ... devem copiar . Uma casinha de porta e janela , em cima duma
        ... coxilha .'''
        >>> for token in text.split(): # doctest: +NORMALIZE_WHITESPACE
        ...     print(st.stem(token))
        clariss risc com giz no quadro-negr a pais que os alun dev copi .
        uma cas de port e janel , em cim dum coxilh .
    