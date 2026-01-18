import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
@MyFunction
def v6_0_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
    H = t_decay.shape[0]
    N = x.shape[-1] // H
    T = x.shape[0]
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    sx = torch.cat((sx.unsqueeze(0), xx[:-1, :])) - xx
    xxx = xx + sx * x_maa
    xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
    xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
    mw, mk, mv, mr, mg = xxx.unbind(dim=0)
    wx = xx + sx * (w_maa + mw)
    kx = xx + sx * (k_maa + mk)
    vx = xx + sx * (v_maa + mv)
    rx = xx + sx * (r_maa + mr)
    gx = xx + sx * (g_maa + mg)
    r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
    g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
    w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)
    return (r, k, v, g, w, xx[-1, :], s.transpose(-1, -2).contiguous())