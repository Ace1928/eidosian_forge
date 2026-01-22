import torch
class MkldnnConv3d(_MkldnnConvNd):

    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(dense_module.weight.to_mkldnn(dtype), self.padding, self.stride, self.dilation, self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(state[0].to_mkldnn(), self.padding, self.stride, self.dilation, self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]