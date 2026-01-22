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
def _abbc(txt):
    """
    abbc = asterisks, breaks, bold, center
    """
    return _center(_bold('<br>' * 10 + '*' * 10 + ' ' + txt + ' ' + '*' * 10))