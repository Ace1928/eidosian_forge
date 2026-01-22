import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
def _collect_one_synset(word, synset, synset_relations):
    """
    Returns the HTML string for one synset or word

    :param word: the current word
    :type word: str
    :param synset: a synset
    :type synset: synset
    :param synset_relations: information about which synset relations
    to display.
    :type synset_relations: dict(synset_key, set(relation_id))
    :return: The HTML string built for this synset
    :rtype: str
    """
    if isinstance(synset, tuple):
        raise NotImplementedError('word not supported by _collect_one_synset')
    typ = 'S'
    pos_tuple = _pos_match((synset.pos(), None, None))
    assert pos_tuple is not None, 'pos_tuple is null: synset.pos(): %s' % synset.pos()
    descr = pos_tuple[2]
    ref = copy.deepcopy(Reference(word, synset_relations))
    ref.toggle_synset(synset)
    synset_label = typ + ';'
    if synset.name() in synset_relations:
        synset_label = _bold(synset_label)
    s = f'<li>{make_lookup_link(ref, synset_label)} ({descr}) '

    def format_lemma(w):
        w = w.replace('_', ' ')
        if w.lower() == word:
            return _bold(w)
        else:
            ref = Reference(w)
            return make_lookup_link(ref, w)
    s += ', '.join((format_lemma(l.name()) for l in synset.lemmas()))
    gl = ' ({}) <i>{}</i> '.format(synset.definition(), '; '.join(('"%s"' % e for e in synset.examples())))
    return s + gl + _synset_relations(word, synset, synset_relations) + '</li>\n'