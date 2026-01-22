from typing import Optional
from typing import Union
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import applications
from tensorflow.keras import layers
from autokeras import keras_layers
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import layer_utils
from autokeras.utils import utils
class RNNBlock(block_module.Block):
    """An RNN Block.

    # Arguments
        return_sequences: Boolean. Whether to return the last output in the
            output sequence, or the full sequence. Defaults to False.
        bidirectional: Boolean or keras_tuner.engine.hyperparameters.Boolean.
            Bidirectional RNN. If left unspecified, it will be
            tuned automatically.
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of layers in RNN. If left unspecified, it will
            be tuned automatically.
        layer_type: String or or keras_tuner.engine.hyperparameters.Choice.
            'gru' or 'lstm'. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(self, return_sequences: bool=False, bidirectional: Optional[Union[bool, hyperparameters.Boolean]]=None, num_layers: Optional[Union[int, hyperparameters.Choice]]=None, layer_type: Optional[Union[str, hyperparameters.Choice]]=None, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.bidirectional = utils.get_hyperparameter(bidirectional, hyperparameters.Boolean('bidirectional', default=True), bool)
        self.num_layers = utils.get_hyperparameter(num_layers, hyperparameters.Choice('num_layers', [1, 2, 3], default=2), int)
        self.layer_type = utils.get_hyperparameter(layer_type, hyperparameters.Choice('layer_type', ['gru', 'lstm'], default='lstm'), str)

    def get_config(self):
        config = super().get_config()
        config.update({'return_sequences': self.return_sequences, 'bidirectional': io_utils.serialize_block_arg(self.bidirectional), 'num_layers': io_utils.serialize_block_arg(self.num_layers), 'layer_type': io_utils.serialize_block_arg(self.layer_type)})
        return config

    @classmethod
    def from_config(cls, config):
        config['bidirectional'] = io_utils.deserialize_block_arg(config['bidirectional'])
        config['num_layers'] = io_utils.deserialize_block_arg(config['num_layers'])
        config['layer_type'] = io_utils.deserialize_block_arg(config['layer_type'])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError('Expect the input tensor of RNNBlock to have dimensions of [batch_size, time_steps, vec_len], but got {shape}'.format(shape=input_node.shape))
        feature_size = shape[-1]
        output_node = input_node
        bidirectional = utils.add_to_hp(self.bidirectional, hp)
        layer_type = utils.add_to_hp(self.layer_type, hp)
        num_layers = utils.add_to_hp(self.num_layers, hp)
        rnn_layers = {'gru': layers.GRU, 'lstm': layers.LSTM}
        in_layer = rnn_layers[layer_type]
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = layers.Bidirectional(in_layer(feature_size, return_sequences=return_sequences))(output_node)
            else:
                output_node = in_layer(feature_size, return_sequences=return_sequences)(output_node)
        return output_node