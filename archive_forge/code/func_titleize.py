import re
import unicodedata
def titleize(word):
    """
    Capitalize all the words and replace some characters in the string to
    create a nicer looking title. :func:`titleize` is meant for creating pretty
    output.

    Examples::

      >>> titleize("man from the boondocks")
      'Man From The Boondocks'
      >>> titleize("x-men: the last stand")
      'X Men: The Last Stand'
      >>> titleize("TheManWithoutAPast")
      'The Man Without A Past'
      >>> titleize("raiders_of_the_lost_ark")
      'Raiders Of The Lost Ark'

    """
    return re.sub("\\b('?\\w)", lambda match: match.group(1).capitalize(), humanize(underscore(word)).title())