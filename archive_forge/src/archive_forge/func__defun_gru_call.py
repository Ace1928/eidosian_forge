import uuid
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_rnn import RNN
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _defun_gru_call(self, inputs, initial_state, training, mask, sequence_lengths):
    self.reset_dropout_mask()
    dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    if dropout_mask is not None:
        inputs = inputs * dropout_mask[0]
    if gru_lstm_utils.use_new_gru_lstm_impl():
        gru_kwargs = {'inputs': inputs, 'init_h': gru_lstm_utils.read_variable_value(initial_state[0]), 'kernel': gru_lstm_utils.read_variable_value(self.cell.kernel), 'recurrent_kernel': gru_lstm_utils.read_variable_value(self.cell.recurrent_kernel), 'bias': gru_lstm_utils.read_variable_value(self.cell.bias), 'mask': mask, 'time_major': self.time_major, 'go_backwards': self.go_backwards, 'sequence_lengths': sequence_lengths, 'zero_output_for_mask': self.zero_output_for_mask}
        last_output, outputs, new_h, runtime = self._defun_wrapper.defun_layer(**gru_kwargs)
    else:
        gpu_gru_kwargs = {'inputs': inputs, 'init_h': gru_lstm_utils.read_variable_value(initial_state[0]), 'kernel': gru_lstm_utils.read_variable_value(self.cell.kernel), 'recurrent_kernel': gru_lstm_utils.read_variable_value(self.cell.recurrent_kernel), 'bias': gru_lstm_utils.read_variable_value(self.cell.bias), 'mask': mask, 'time_major': self.time_major, 'go_backwards': self.go_backwards, 'sequence_lengths': sequence_lengths, 'return_sequences': self.return_sequences}
        normal_gru_kwargs = gpu_gru_kwargs.copy()
        normal_gru_kwargs.update({'zero_output_for_mask': self.zero_output_for_mask})
        if tf.executing_eagerly():
            device_type = gru_lstm_utils.get_context_device_type()
            can_use_gpu = (device_type == gru_lstm_utils.GPU_DEVICE_NAME or (device_type is None and tf.config.list_logical_devices('GPU'))) and gru_lstm_utils.is_cudnn_supported_inputs(mask, self.time_major, sequence_lengths)
            if can_use_gpu:
                last_output, outputs, new_h, runtime = gpu_gru(**gpu_gru_kwargs)
            else:
                last_output, outputs, new_h, runtime = standard_gru(**normal_gru_kwargs)
        else:
            last_output, outputs, new_h, runtime = gru_with_backend_selection(**normal_gru_kwargs)
    states = [new_h]
    return (last_output, outputs, runtime, states)