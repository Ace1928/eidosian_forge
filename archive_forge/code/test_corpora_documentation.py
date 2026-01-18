from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
Test with non-trivial directory structure, shown below:
        .
        ├── 0.txt
        ├── a_folder
        │   └── 1.txt
        └── b_folder
            ├── 2.txt
            ├── 3.txt
            └── c_folder
                └── 4.txt
        