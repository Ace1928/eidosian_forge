import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class MobileV2PlusRNNModel(RecurrentNetwork):
    """A conv. + recurrent keras net example using a pre-trained MobileNet."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, cnn_shape):
        super(MobileV2PlusRNNModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = 16
        visual_size = cnn_shape[0] * cnn_shape[1] * cnn_shape[2]
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='h')
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='c')
        seq_in = tf.keras.layers.Input(shape=(), name='seq_in', dtype=tf.int32)
        inputs = tf.keras.layers.Input(shape=(None, visual_size), name='visual_inputs')
        input_visual = inputs
        input_visual = tf.reshape(input_visual, [-1, cnn_shape[0], cnn_shape[1], cnn_shape[2]])
        cnn_input = tf.keras.layers.Input(shape=cnn_shape, name='cnn_input')
        cnn_model = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, include_top=True, weights=None, input_tensor=cnn_input, pooling=None)
        vision_out = cnn_model(input_visual)
        vision_out = tf.reshape(vision_out, [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(self.cell_size, return_sequences=True, return_state=True, name='lstm')(inputs=vision_out, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c])
        logits = tf.keras.layers.Dense(self.num_outputs, activation=tf.keras.activations.linear, name='logits')(lstm_out)
        values = tf.keras.layers.Dense(1, activation=None, name='values')(lstm_out)
        self.rnn_model = tf.keras.Model(inputs=[inputs, seq_in, state_in_h, state_in_c], outputs=[logits, values, state_h, state_c])
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return (model_out, [h, c])

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])