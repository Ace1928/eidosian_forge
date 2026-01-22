from typing import Optional
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
class AsteroidISTFT(nn.Module):

    def __init__(self, fb):
        super(AsteroidISTFT, self).__init__()
        self.dec = Decoder(fb)

    def forward(self, X: Tensor, length: Optional[int]=None) -> Tensor:
        aux = from_torchaudio(X)
        return self.dec(aux, length=length)