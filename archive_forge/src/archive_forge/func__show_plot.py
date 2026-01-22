import codecs
import csv
import json
import pickle
import random
import re
import sys
import time
from copy import deepcopy
import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.data import load
from nltk.tokenize.casual import EMOTICON_RE
def _show_plot(x_values, y_values, x_labels=None, y_labels=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('The plot function requires matplotlib to be installed.See https://matplotlib.org/') from e
    plt.locator_params(axis='y', nbins=3)
    axes = plt.axes()
    axes.yaxis.grid()
    plt.plot(x_values, y_values, 'ro', color='red')
    plt.ylim(ymin=-1.2, ymax=1.2)
    plt.tight_layout(pad=5)
    if x_labels:
        plt.xticks(x_values, x_labels, rotation='vertical')
    if y_labels:
        plt.yticks([-1, 0, 1], y_labels, rotation='horizontal')
    plt.margins(0.2)
    plt.show()