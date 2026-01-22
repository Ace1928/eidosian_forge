from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
def _ThreeLetterGenerator(validate):
    """Generate random 3-letter words.

  Words are generated in a consonant-vowel-consonant order to be pronounceable.
  A specific word matching this pattern has a 1/21*1/5*1/21 = 1/2205 chance
  of being generated.

  Args:
    validate: bool, True to validate words against the invalid set.

  Returns:
    str, 3-letter word
  """
    while True:
        word = random.choice(_CONSONANTS) + random.choice(_VOWELS) + random.choice(_CONSONANTS)
        if not validate or IsValidWord(word):
            return word