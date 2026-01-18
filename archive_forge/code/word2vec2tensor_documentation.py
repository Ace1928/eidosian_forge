import os
import sys
import logging
import argparse
import gensim
from gensim import utils
Convert file in Word2Vec format and writes two files 2D tensor TSV file.

    File "tensor_filename"_tensor.tsv contains word-vectors, "tensor_filename"_metadata.tsv contains words.

    Parameters
    ----------
    word2vec_model_path : str
        Path to file in Word2Vec format.
    tensor_filename : str
        Prefix for output files.
    binary : bool, optional
        True if input file in binary format.

    