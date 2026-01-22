import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _parse_save_args(ext: Optional[str], format: Optional[str], encoding: Optional[str], bps: Optional[int]):

    def _type(spec):
        return format == spec or (format is None and ext == spec)
    if _type('wav') or _type('amb'):
        muxer = 'wav'
        encoder = _get_encoder_for_wav(encoding, bps)
        sample_fmt = None
    elif _type('vorbis'):
        muxer = 'ogg'
        encoder = 'vorbis'
        sample_fmt = None
    else:
        muxer = format
        encoder = None
        sample_fmt = None
        if _type('flac'):
            sample_fmt = _get_flac_sample_fmt(bps)
        if _type('ogg'):
            sample_fmt = _get_flac_sample_fmt(bps)
    return (muxer, encoder, sample_fmt)