from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer
@staticmethod
def _lemmatize(word):
    """
        Use morphy from WordNet to find the base form of verbs.
        """
    from nltk.corpus import wordnet as wn
    lemma = wn.morphy(word, pos=wn.VERB)
    if lemma is not None:
        return lemma
    return word