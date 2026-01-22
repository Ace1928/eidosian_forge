import warnings
from typing import Optional, Tuple
import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData
def _get_subtype(dtype: torch.dtype, format: str, encoding: str, bits_per_sample: int):
    if format == 'wav':
        return _get_subtype_for_wav(dtype, encoding, bits_per_sample)
    if format == 'flac':
        if encoding:
            raise ValueError('flac does not support encoding.')
        if not bits_per_sample:
            return 'PCM_16'
        if bits_per_sample > 24:
            raise ValueError('flac does not support bits_per_sample > 24.')
        return 'PCM_S8' if bits_per_sample == 8 else f'PCM_{bits_per_sample}'
    if format in ('ogg', 'vorbis'):
        if bits_per_sample:
            raise ValueError('ogg/vorbis does not support bits_per_sample.')
        if encoding is None or encoding == 'vorbis':
            return 'VORBIS'
        if encoding == 'opus':
            return 'OPUS'
        raise ValueError(f'Unexpected encoding: {encoding}')
    if format == 'mp3':
        return 'MPEG_LAYER_III'
    if format == 'sph':
        return _get_subtype_for_sphere(encoding, bits_per_sample)
    if format in ('nis', 'nist'):
        return 'PCM_16'
    raise ValueError(f'Unsupported format: {format}')