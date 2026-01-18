import logging
import os.path
import sys
from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel

USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This actually creates several files:

* `OUTPUT_PREFIX_wordids.txt.bz2`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation in Matrix Market format
* `OUTPUT_PREFIX_bow.mm.index`: index for `OUTPUT_PREFIX_bow.mm`
* `OUTPUT_PREFIX_bow.mm.metadata.cpickle`: titles of documents
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation in Matrix Market format
* `OUTPUT_PREFIX_tfidf.mm.index`: index for `OUTPUT_PREFIX_tfidf.mm`
* `OUTPUT_PREFIX.tfidf_model`: TF-IDF model

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
100,000.

Example:
  python -m gensim.scripts.make_wikicorpus ~/gensim/results/enwiki-latest-pages-articles.xml.bz2 ~/gensim/results/wiki
