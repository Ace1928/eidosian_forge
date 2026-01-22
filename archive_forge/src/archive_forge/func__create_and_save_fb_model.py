from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def _create_and_save_fb_model(fname, model_params):
    model = FT_gensim(**model_params)
    lee_data = LineSentence(datapath('lee_background.cor'))
    model.build_vocab(lee_data)
    model.train(lee_data, total_examples=model.corpus_count, epochs=model.epochs)
    gensim.models.fasttext.save_facebook_model(model, fname)
    return model