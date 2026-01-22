from torch import nn
class QuantWrapper(nn.Module):
    """A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    def __init__(self, module):
        super().__init__()
        qconfig = getattr(module, 'qconfig', None)
        self.add_module('quant', QuantStub(qconfig))
        self.add_module('dequant', DeQuantStub(qconfig))
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)