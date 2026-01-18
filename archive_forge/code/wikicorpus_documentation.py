import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
Iterate over the dump, yielding a list of tokens for each article that passed
        the length and namespace filtering.

        Uses multiprocessing internally to parallelize the work and process the dump more quickly.

        Notes
        -----
        This iterates over the **texts**. If you want vectors, just use the standard corpus interface
        instead of this method:

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.corpora import WikiCorpus
            >>>
            >>> path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
            >>>
            >>> for vec in WikiCorpus(path_to_wiki_dump):
            ...     pass

        Yields
        ------
        list of str
            If `metadata` is False, yield only list of token extracted from the article.
        (list of str, (int, str))
            List of tokens (extracted from the article), page id and article title otherwise.

        