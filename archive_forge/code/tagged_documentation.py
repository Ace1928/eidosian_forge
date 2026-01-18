import os
from nltk.corpus.reader.api import *
from nltk.corpus.reader.timit import read_timit_block
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
from nltk.tokenize import *
Reads one paragraph at a time.