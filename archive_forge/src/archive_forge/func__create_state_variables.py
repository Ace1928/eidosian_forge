import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
@tracking.no_automatic_dependency_tracking
def _create_state_variables(self, batch_size):
    with backend.name_scope(self.name, caller=self):
        self.states = tree.map_structure(lambda value: backend.Variable(value, trainable=False, dtype=self.variable_dtype, name='rnn_state'), self.get_initial_state(batch_size))