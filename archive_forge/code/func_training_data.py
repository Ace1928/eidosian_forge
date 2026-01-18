import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.fixture(scope='session')
def training_data():
    return [['a', 'b', 'c', 'd'], ['e', 'g', 'a', 'd', 'b', 'e']]