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
def _synset_relations(word, synset, synset_relations):
    """
    Builds the HTML string for the relations of a synset

    :param word: The current word
    :type word: str
    :param synset: The synset for which we're building the relations.
    :type synset: Synset
    :param synset_relations: synset keys and relation types for which to display relations.
    :type synset_relations: dict(synset_key, set(relation_type))
    :return: The HTML for a synset's relations
    :rtype: str
    """
    if not synset.name() in synset_relations:
        return ''
    ref = Reference(word, synset_relations)

    def relation_html(r):
        if isinstance(r, Synset):
            return make_lookup_link(Reference(r.lemma_names()[0]), r.lemma_names()[0])
        elif isinstance(r, Lemma):
            return relation_html(r.synset())
        elif isinstance(r, tuple):
            return '{}\n<ul>{}</ul>\n'.format(relation_html(r[0]), ''.join(('<li>%s</li>\n' % relation_html(sr) for sr in r[1])))
        else:
            raise TypeError('r must be a synset, lemma or list, it was: type(r) = %s, r = %s' % (type(r), r))

    def make_synset_html(db_name, disp_name, rels):
        synset_html = '<i>%s</i>\n' % make_lookup_link(copy.deepcopy(ref).toggle_synset_relation(synset, db_name), disp_name)
        if db_name in ref.synset_relations[synset.name()]:
            synset_html += '<ul>%s</ul>\n' % ''.join(('<li>%s</li>\n' % relation_html(r) for r in rels))
        return synset_html
    html = '<ul>' + '\n'.join(('<li>%s</li>' % make_synset_html(*rel_data) for rel_data in get_relations_data(word, synset) if rel_data[2] != [])) + '</ul>'
    return html