import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
class SwitchboardTurn(list):
    """
    A specialized list object used to encode switchboard utterances.
    The elements of the list are the words in the utterance; and two
    attributes, ``speaker`` and ``id``, are provided to retrieve the
    spearker identifier and utterance id.  Note that utterance ids
    are only unique within a given discourse.
    """

    def __init__(self, words, speaker, id):
        list.__init__(self, words)
        self.speaker = speaker
        self.id = int(id)

    def __repr__(self):
        if len(self) == 0:
            text = ''
        elif isinstance(self[0], tuple):
            text = ' '.join(('%s/%s' % w for w in self))
        else:
            text = ' '.join(self)
        return f'<{self.speaker}.{self.id}: {text!r}>'