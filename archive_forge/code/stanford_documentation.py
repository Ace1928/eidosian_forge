import json
import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_jar, java
from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize.api import TokenizerI

        Use stanford tokenizer's PTBTokenizer to tokenize multiple sentences.
        