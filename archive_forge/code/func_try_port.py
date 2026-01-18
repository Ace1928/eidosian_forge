import json
import os  # required for doctests
import re
import socket
import time
from typing import List, Tuple
from nltk.internals import _java_options, config_java, find_jar_iter, java
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tag.api import TaggerI
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree
def try_port(port=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', port))
    p = sock.getsockname()[1]
    sock.close()
    return p