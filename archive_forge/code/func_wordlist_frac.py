import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def wordlist_frac(utt, history, word_list):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that are in word_list.
    Additional inputs:
      word_list: list of strings.
    """
    words = utt.split()
    num_in_list = len([w for w in words if w in word_list])
    return num_in_list / len(words)