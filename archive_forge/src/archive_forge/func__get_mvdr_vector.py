import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F
def _get_mvdr_vector(psd_s: torch.Tensor, psd_n: torch.Tensor, reference_vector: torch.Tensor, solution: str='ref_channel', diagonal_loading: bool=True, diag_eps: float=1e-07, eps: float=1e-08) -> torch.Tensor:
    """Compute the MVDR beamforming weights with ``solution`` argument.

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_vector (torch.Tensor): one-hot reference channel matrix.
        solution (str, optional): Solution to compute the MVDR beamforming weights.
            Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        eps (float, optional): Value to add to the denominator in the beamforming weight formula.
            (Default: ``1e-8``)

    Returns:
        torch.Tensor: the mvdr beamforming weight matrix
    """
    if solution == 'ref_channel':
        beamform_vector = F.mvdr_weights_souden(psd_s, psd_n, reference_vector, diagonal_loading, diag_eps, eps)
    else:
        if solution == 'stv_evd':
            stv = F.rtf_evd(psd_s)
        else:
            stv = F.rtf_power(psd_s, psd_n, reference_vector, diagonal_loading=diagonal_loading, diag_eps=diag_eps)
        beamform_vector = F.mvdr_weights_rtf(stv, psd_n, reference_vector, diagonal_loading, diag_eps, eps)
    return beamform_vector