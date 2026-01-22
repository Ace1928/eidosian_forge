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

        Tag multiple sentences.

        Takes multiple sentences as a list where each sentence is a string.

        :param sentences: Input sentences to tag
        :type sentences: list(str)
        :rtype: list(list(list(tuple(str, str)))
        