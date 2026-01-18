import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll

        Train MaltParser from a file
        :param conll_file: str for the filename of the training input data
        :type conll_file: str
        