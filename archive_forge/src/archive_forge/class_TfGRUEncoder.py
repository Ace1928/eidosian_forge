from typing import Optional
import tree  # pip install dm_tree
from ray.rllib.core.models.base import (
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.tf.base import TfModel
from ray.rllib.core.models.tf.primitives import TfMLP, TfCNN
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
class TfGRUEncoder(TfModel, Encoder):
    """A recurrent GRU encoder.

    This encoder has...
    - Zero or one tokenizers.
    - One or more GRU layers.
    - One linear output layer.
    """

    def __init__(self, config: RecurrentEncoderConfig) -> None:
        TfModel.__init__(self, config)
        if config.tokenizer_config is not None:
            self.tokenizer = config.tokenizer_config.build(framework='tf2')
            input_dims = (1,) + tuple(self.tokenizer.output_specs[ENCODER_OUT].full_shape)
        else:
            self.tokenizer = None
            input_dims = (1, 1) + tuple(config.input_dims)
        self.grus = []
        for _ in range(config.num_layers):
            layer = tf.keras.layers.GRU(config.hidden_dim, time_major=not config.batch_major, use_bias=config.use_bias, return_sequences=True, return_state=True)
            layer.build(input_dims)
            input_dims = (1, 1, config.hidden_dim)
            self.grus.append(layer)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict({SampleBatch.OBS: TensorSpec('b, t, d', d=self.config.input_dims[0], framework='tf2'), STATE_IN: {'h': TensorSpec('b, l, h', h=self.config.hidden_dim, l=self.config.num_layers, framework='tf2')}})

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict({ENCODER_OUT: TensorSpec('b, t, d', d=self.config.output_dims[0], framework='tf2'), STATE_OUT: {'h': TensorSpec('b, l, h', h=self.config.hidden_dim, l=self.config.num_layers, framework='tf2')}})

    @override(Model)
    def get_initial_state(self):
        return {'h': tf.zeros((self.config.num_layers, self.config.hidden_dim))}

    @override(Model)
    def _forward(self, inputs: NestedDict, **kwargs) -> NestedDict:
        outputs = {}
        if self.tokenizer is not None:
            out = tokenize(self.tokenizer, inputs, framework='tf2')
        else:
            out = tf.cast(inputs[SampleBatch.OBS], tf.float32)
        states_in = tree.map_structure(lambda s: tf.transpose(s, perm=[1, 0] + list(range(2, len(s.shape)))), inputs[STATE_IN])
        states_out = []
        for i, layer in enumerate(self.grus):
            out, h = layer(out, states_in['h'][i])
            states_out.append(h)
        outputs[ENCODER_OUT] = out
        outputs[STATE_OUT] = {'h': tf.stack(states_out, 1)}
        return outputs