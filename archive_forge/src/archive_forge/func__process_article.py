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
def _process_article(args):
    """Same as :func:`~gensim.corpora.wikicorpus.process_article`, but with args in list format.

    Parameters
    ----------
    args : [(str, bool, str, int), (function, int, int, bool)]
        First element - same as `args` from :func:`~gensim.corpora.wikicorpus.process_article`,
        second element is tokenizer function, token minimal length, token maximal length, lowercase flag.

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    Warnings
    --------
    Should not be called explicitly. Use :func:`~gensim.corpora.wikicorpus.process_article` instead.

    """
    tokenizer_func, token_min_len, token_max_len, lower = args[-1]
    args = args[:-1]
    return process_article(args, tokenizer_func=tokenizer_func, token_min_len=token_min_len, token_max_len=token_max_len, lower=lower)