import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _maybe_config_dropout_masks(self, cell, input_sequence, input_state):
    step_input = input_sequence[:, 0, :]
    state = input_state[0] if isinstance(input_state, (list, tuple)) else input_state
    if isinstance(cell, DropoutRNNCell):
        cell.get_dropout_mask(step_input)
        cell.get_recurrent_dropout_mask(state)
    if isinstance(cell, StackedRNNCells):
        for c, s in zip(cell.cells, input_state):
            self._maybe_config_dropout_masks(c, input_sequence, s)