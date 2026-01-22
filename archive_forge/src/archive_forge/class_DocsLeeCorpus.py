from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
class DocsLeeCorpus:

    def __init__(self, string_tags=False, unicode_tags=False):
        self.string_tags = string_tags
        self.unicode_tags = unicode_tags

    def _tag(self, i):
        if self.unicode_tags:
            return u'_¡_%d' % i
        elif self.string_tags:
            return '_*%d' % i
        return i

    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for i, line in enumerate(f):
                yield doc2vec.TaggedDocument(utils.simple_preprocess(line), [self._tag(i)])