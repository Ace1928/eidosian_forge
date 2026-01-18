import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist

        Returns the ARFF data section for the given data.

        :param tokens: a list of featuresets (dicts) or labelled featuresets
            which are tuples (featureset, label).
        :param labeled: Indicates whether the given tokens are labeled
            or not.  If None, then the tokens will be assumed to be
            labeled if the first token's value is a tuple or list.
        