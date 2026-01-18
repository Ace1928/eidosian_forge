from collections import OrderedDict
from torch import nn
from vllm.utils import LRUCache
from vllm.lora.utils import (parse_fine_tuned_lora_name, replace_submodule)
def test_replace_submodule():
    model = nn.Sequential(OrderedDict([('dense1', nn.Linear(764, 100)), ('act1', nn.ReLU()), ('dense2', nn.Linear(100, 50)), ('seq1', nn.Sequential(OrderedDict([('dense1', nn.Linear(100, 10)), ('dense2', nn.Linear(10, 50))]))), ('act2', nn.ReLU()), ('output', nn.Linear(50, 10)), ('outact', nn.Sigmoid())]))
    sigmoid = nn.Sigmoid()
    replace_submodule(model, 'act1', sigmoid)
    assert dict(model.named_modules())['act1'] == sigmoid
    dense2 = nn.Linear(1, 5)
    replace_submodule(model, 'seq1.dense2', dense2)
    assert dict(model.named_modules())['seq1.dense2'] == dense2