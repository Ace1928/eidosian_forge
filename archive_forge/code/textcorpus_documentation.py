from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import (
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
Calculate length of corpus and cache it to `self.length`.