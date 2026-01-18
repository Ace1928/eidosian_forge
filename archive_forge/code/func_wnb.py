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
def wnb(port=8000, runBrowser=True, logfilename=None):
    """
    Run NLTK Wordnet Browser Server.

    :param port: The port number for the server to listen on, defaults to
                 8000
    :type  port: int

    :param runBrowser: True to start a web browser and point it at the web
                       server.
    :type  runBrowser: bool
    """
    global server_mode, logfile
    server_mode = not runBrowser
    if logfilename:
        try:
            logfile = open(logfilename, 'a', 1)
        except OSError as e:
            sys.stderr.write("Couldn't open %s for writing: %s", logfilename, e)
            sys.exit(1)
    else:
        logfile = None
    url = 'http://localhost:' + str(port)
    server_ready = None
    browser_thread = None
    if runBrowser:
        server_ready = threading.Event()
        browser_thread = startBrowser(url, server_ready)
    server = HTTPServer(('', port), MyServerHandler)
    if logfile:
        logfile.write('NLTK Wordnet browser server running serving: %s\n' % url)
    if runBrowser:
        server_ready.set()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if runBrowser:
        browser_thread.join()
    if logfile:
        logfile.close()