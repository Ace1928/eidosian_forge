import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.fixture(scope='session')
def trigram_training_data(training_data):
    return [list(padded_everygrams(3, sent)) for sent in training_data]