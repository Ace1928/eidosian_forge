import logging
import unittest
import os
import zlib
from gensim.corpora.hashdictionary import HashDictionary
from gensim.test.utils import get_tmpfile, common_texts
 `HashDictionary` can be saved & loaded as compressed pickle. 