import logging
import os.path
import sys
import argparse
from numpy import seterr
from gensim.models.word2vec import Word2Vec, LineSentence  # avoid referencing __main__ in pickle

USAGE: %(program)s -train CORPUS -output VECTORS -size SIZE -window WINDOW
-cbow CBOW -sample SAMPLE -hs HS -negative NEGATIVE -threads THREADS -iter ITER
-min_count MIN-COUNT -alpha ALPHA -binary BINARY -accuracy FILE

Trains a neural embedding model on text file CORPUS.
Parameters essentially reproduce those used by the original C tool
(see https://code.google.com/archive/p/word2vec/).

Parameters for training:
        -train <file>
                Use text data from <file> to train the model
        -output <file>
                Use <file> to save the resulting word vectors / word clusters
        -size <int>
                Set size of word vectors; default is 100
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -hs <int>
                Use Hierarchical Softmax; default is 0 (not used)
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
                Use <int> threads (default 3)
        -iter <int>
                Run more training iterations (default 5)
        -min_count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        -binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
        -cbow <int>
                Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
        -accuracy <file>
                Compute accuracy of the resulting model analogical inference power on questions file <file>
                See an example of questions file
                at https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt

Example: python -m gensim.scripts.word2vec_standalone -train data.txt          -output vec.txt -size 200 -sample 1e-4 -binary 0 -iter 3
