import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
Recover something like neural net layers from PyTorch Module's and the
        compute graph from a Variable.

        Example output for a multi-layer RNN. We confusingly assign shared embedding values
        to the encoder, but ordered next to the decoder.

        rnns.0.linear.module.weight_raw rnns.0
        rnns.0.linear.module.bias rnns.0
        rnns.1.linear.module.weight_raw rnns.1
        rnns.1.linear.module.bias rnns.1
        rnns.2.linear.module.weight_raw rnns.2
        rnns.2.linear.module.bias rnns.2
        rnns.3.linear.module.weight_raw rnns.3
        rnns.3.linear.module.bias rnns.3
        decoder.weight encoder
        decoder.bias decoder
        