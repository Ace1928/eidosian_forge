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
def toggle_synset(self, synset):
    """
        Toggle displaying of the relation types for the given synset
        """
    if synset.name() in self.synset_relations:
        del self.synset_relations[synset.name()]
    else:
        self.synset_relations[synset.name()] = set()
    return self