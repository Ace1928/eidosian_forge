from enchant.errors import Error
import locale
def trim_suggestions(word, suggs, maxlen, calcdist=None):
    """Trim a list of suggestions to a maximum length.

    If the list of suggested words is too long, you can use this function
    to trim it down to a maximum length.  It tries to keep the "best"
    suggestions based on similarity to the original word.

    If the optional "calcdist" argument is provided, it must be a callable
    taking two words and returning the distance between them.  It will be
    used to determine which words to retain in the list.  The default is
    a simple Levenshtein distance.
    """
    if calcdist is None:
        calcdist = levenshtein
    decorated = [(calcdist(word, s), s) for s in suggs]
    decorated.sort()
    return [s for l, s in decorated[:maxlen]]