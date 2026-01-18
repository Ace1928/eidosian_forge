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
def toggle_synset_relation(self, synset, relation):
    """
        Toggle the display of the relations for the given synset and
        relation type.

        This function will throw a KeyError if the synset is currently
        not being displayed.
        """
    if relation in self.synset_relations[synset.name()]:
        self.synset_relations[synset.name()].remove(relation)
    else:
        self.synset_relations[synset.name()].add(relation)
    return self