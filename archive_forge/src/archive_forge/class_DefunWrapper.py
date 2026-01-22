import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
class DefunWrapper:
    """A wrapper with no deep copy of the Defun in LSTM/GRU layer."""

    def __init__(self, time_major, go_backwards, layer_name):
        self.time_major = time_major
        self.go_backwards = go_backwards
        self.layer_name = layer_name
        if self.layer_name not in ['lstm', 'gru']:
            raise ValueError('Defun wrapper only applies to LSTM and GRU layer, but given {}'.format(self.layer_name))
        supportive_attributes = {'time_major': self.time_major, 'go_backwards': self.go_backwards, _FUNCTION_API_NAME_ATTRIBUTE: self.layer_name + '_' + str(uuid.uuid4())}
        if self.layer_name == 'lstm':
            from keras.src.layers.rnn import lstm
            layer_func = lstm.lstm_with_backend_selection
        else:
            from keras.src.layers.rnn import gru
            layer_func = gru.gru_with_backend_selection
        self.defun_layer = tf.function(layer_func, autograph=False, experimental_attributes=supportive_attributes)

    def __deepcopy__(self, memo):
        new_wrapper = type(self)(self.time_major, self.go_backwards, self.layer_name)
        memo[id(self)] = new_wrapper
        return new_wrapper