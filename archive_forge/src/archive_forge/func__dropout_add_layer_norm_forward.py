import dropout_layer_norm
import torch
from torch.nn import init
def _dropout_add_layer_norm_forward(x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32=False, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(x0mat, residualmat, gamma, beta, rowscale, colscale, None, None, dropout_p, epsilon, 1.0, 0, None, residual_in_fp32, is_rms_norm)
    return (zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma)