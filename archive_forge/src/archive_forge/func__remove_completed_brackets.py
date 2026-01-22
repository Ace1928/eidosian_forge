import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _remove_completed_brackets(self):

    def _bracket_is_incomplete(bracket):
        bracket_num = bracket['bracket_num']
        rounds = bracket['rounds']
        last_round = len(rounds) - 1
        return len(rounds[last_round]) != self._get_size(bracket_num, last_round)
    self._brackets = list(filter(_bracket_is_incomplete, self._brackets))