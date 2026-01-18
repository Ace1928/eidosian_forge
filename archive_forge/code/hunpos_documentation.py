import os
from subprocess import PIPE, Popen
from nltk.internals import find_binary, find_file
from nltk.tag.api import TaggerI
Tags a single sentence: a list of words.
        The tokens should not contain any newline characters.
        