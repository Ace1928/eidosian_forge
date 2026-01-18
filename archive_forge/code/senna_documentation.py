from os import environ, path, sep
from platform import architecture, system
from subprocess import PIPE, Popen
from nltk.tag.api import TaggerI

        Applies the tag method over a list of sentences. This method will return a
        list of dictionaries. Every dictionary will contain a word with its
        calculated annotations/tags.
        