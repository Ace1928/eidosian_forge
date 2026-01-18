import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple

    A specialized list object used to encode switchboard utterances.
    The elements of the list are the words in the utterance; and two
    attributes, ``speaker`` and ``id``, are provided to retrieve the
    spearker identifier and utterance id.  Note that utterance ids
    are only unique within a given discourse.
    