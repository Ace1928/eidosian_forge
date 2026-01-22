import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def _forward_unpadded(self, x, x_mask):
    """
        Faster encoding that ignores any padding.
        """
    x = x.transpose(0, 1).contiguous()
    outputs = [x]
    for i in range(self.num_layers):
        rnn_input = outputs[-1]
        if self.dropout_rate > 0:
            rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
        rnn_output = self.rnns[i](rnn_input)[0]
        outputs.append(rnn_output)
    if self.concat_layers:
        output = torch.cat(outputs[1:], 2)
    else:
        output = outputs[-1]
    output = output.transpose(0, 1).contiguous()
    if self.dropout_output and self.dropout_rate > 0:
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
    return output