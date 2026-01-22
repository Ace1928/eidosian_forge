import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _pretty_fulltext_sentences(sents):
    """
    Helper function for pretty-printing a list of annotated sentences for a full-text document.

    :param sent: The list of sentences to be printed.
    :type sent: list(AttrDict)
    :return: An index of the text of the sentences.
    :rtype: str
    """
    outstr = ''
    outstr += 'full-text document ({0.ID}) {0.name}:\n\n'.format(sents)
    outstr += '[corpid] {0.corpid}\n[corpname] {0.corpname}\n[description] {0.description}\n[URL] {0.URL}\n\n'.format(sents)
    outstr += f'[sentence]\n'
    for i, sent in enumerate(sents.sentence):
        outstr += f'[{i}] {sent.text}\n'
    outstr += '\n'
    return outstr