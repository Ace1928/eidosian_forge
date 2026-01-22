import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class TorchMobileV2PlusRNNModel(TorchRNN, nn.Module):
    """A conv. + recurrent torch net example using a pre-trained MobileNet."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, cnn_shape):
        TorchRNN.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.lstm_state_size = 16
        self.cnn_shape = list(cnn_shape)
        self.visual_size_in = cnn_shape[0] * cnn_shape[1] * cnn_shape[2]
        self.visual_size_out = 1000
        self.cnn_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.lstm = nn.LSTM(self.visual_size_out, self.lstm_state_size, batch_first=True)
        self.logits = SlimFC(self.lstm_state_size, self.num_outputs)
        self.value_branch = SlimFC(self.lstm_state_size, 1)
        self._features = None

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        vision_in = torch.reshape(inputs, [-1] + self.cnn_shape)
        vision_out = self.cnn_model(vision_in)
        vision_out_time_ranked = torch.reshape(vision_out, [inputs.shape[0], inputs.shape[1], vision_out.shape[-1]])
        if len(state[0].shape) == 2:
            state[0] = state[0].unsqueeze(0)
            state[1] = state[1].unsqueeze(0)
        self._features, [h, c] = self.lstm(vision_out_time_ranked, state)
        logits = self.logits(self._features)
        return (logits, [h.squeeze(0), c.squeeze(0)])

    @override(ModelV2)
    def get_initial_state(self):
        h = [list(self.cnn_model.modules())[-1].weight.new(1, self.lstm_state_size).zero_().squeeze(0), list(self.cnn_model.modules())[-1].weight.new(1, self.lstm_state_size).zero_().squeeze(0)]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, 'must call forward() first'
        return torch.reshape(self.value_branch(self._features), [-1])