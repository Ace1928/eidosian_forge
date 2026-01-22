import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.fixture
def absolute_discounting_trigram_model(trigram_training_data, vocabulary):
    model = AbsoluteDiscountingInterpolated(order=3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model