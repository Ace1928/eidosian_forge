import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class _LPIPS(nn.Module):

    def __init__(self, pretrained: bool=True, net: Literal['alex', 'vgg', 'squeeze']='alex', spatial: bool=False, pnet_rand: bool=False, pnet_tune: bool=False, use_dropout: bool=True, model_path: Optional[str]=None, eval_mode: bool=True, resize: Optional[int]=None) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            pretrained: This flag controls the linear layers should be pretrained version or random
            net: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            spatial: If input should be spatial averaged
            pnet_rand: If backbone should be random or use imagenet pre-trained weights
            pnet_tune: If backprop should be enabled for both backbone and linear layers
            use_dropout: If dropout layers should be added
            model_path: Model path to load pretained models from
            eval_mode: If network should be in evaluation mode
            resize: If input should be resized to this size

        """
        super().__init__()
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.resize = resize
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = Alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = SqueezeNet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == 'squeeze':
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)
        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', f'lpips_models/{net}.pth'))
            self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        if eval_mode:
            self.eval()
        if not self.pnet_tune:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, in0: Tensor, in1: Tensor, retperlayer: bool=False, normalize: bool=False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)
        outs0, outs1 = (self.net.forward(in0_input), self.net.forward(in1_input))
        feats0, feats1, diffs = ({}, {}, {})
        for kk in range(self.L):
            feats0[kk], feats1[kk] = (_normalize_tensor(outs0[kk]), _normalize_tensor(outs1[kk]))
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = []
        for kk in range(self.L):
            if self.spatial:
                res.append(_upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:])))
            else:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))
        val: Tensor = sum(res)
        if retperlayer:
            return (val, res)
        return val