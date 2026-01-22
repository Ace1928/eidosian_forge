import os
from typing import Optional, Tuple
import torch
import torchaudio
from torchaudio import AudioMetaData
Save audio data to file.

    Args:
        filepath (path-like object): Path to save file.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float or None, optional): Used for formats other than WAV.
            This corresponds to ``-C`` option of ``sox`` command.

            ``"mp3"``
                Either bitrate (in ``kbps``) with quality factor, such as ``128.2``, or
                VBR encoding with quality factor such as ``-4.2``. Default: ``-4.5``.

            ``"flac"``
                Whole number from ``0`` to ``8``. ``8`` is default and highest compression.

            ``"ogg"``, ``"vorbis"``
                Number from ``-1`` to ``10``; ``-1`` is the highest compression
                and lowest quality. Default: ``3``.

            See the detail at http://sox.sourceforge.net/soxformat.html.
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is infered from
            file extension. If file extension is missing or different, you can specify the
            correct format with this argument.

            When ``filepath`` argument is file-like object, this argument is required.

            Valid values are ``"wav"``, ``"mp3"``, ``"ogg"``, ``"vorbis"``, ``"amr-nb"``,
            ``"amb"``, ``"flac"``, ``"sph"``, ``"gsm"``, and ``"htk"``.

        encoding (str or None, optional): Changes the encoding for the supported formats.
            This argument is effective only for supported formats, such as ``"wav"``, ``""amb"``
            and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

            Default values
                If not provided, the default value is picked based on ``format`` and ``bits_per_sample``.

                ``"wav"``, ``"amb"``
                    - | If both ``encoding`` and ``bits_per_sample`` are not provided, the ``dtype`` of the
                      | Tensor is used to determine the default value.

                        - ``"PCM_U"`` if dtype is ``uint8``
                        - ``"PCM_S"`` if dtype is ``int16`` or ``int32``
                        - ``"PCM_F"`` if dtype is ``float32``

                    - ``"PCM_U"`` if ``bits_per_sample=8``
                    - ``"PCM_S"`` otherwise

                ``"sph"`` format;
                    - the default value is ``"PCM_S"``

        bits_per_sample (int or None, optional): Changes the bit depth for the supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"``, ``"sph"``, or ``"amb"``, you can change the
            bit depth. Valid values are ``8``, ``16``, ``32`` and ``64``.

            Default Value;
                If not provided, the default values are picked based on ``format`` and ``"encoding"``;

                ``"wav"``, ``"amb"``;
                    - | If both ``encoding`` and ``bits_per_sample`` are not provided, the ``dtype`` of the
                      | Tensor is used.

                        - ``8`` if dtype is ``uint8``
                        - ``16`` if dtype is ``int16``
                        - ``32`` if dtype is  ``int32`` or ``float32``

                    - ``8`` if ``encoding`` is ``"PCM_U"``, ``"ULAW"`` or ``"ALAW"``
                    - ``16`` if ``encoding`` is ``"PCM_S"``
                    - ``32`` if ``encoding`` is ``"PCM_F"``

                ``"flac"`` format;
                    - the default value is ``24``

                ``"sph"`` format;
                    - ``16`` if ``encoding`` is ``"PCM_U"``, ``"PCM_S"``, ``"PCM_F"`` or not provided.
                    - ``8`` if ``encoding`` is ``"ULAW"`` or ``"ALAW"``

                ``"amb"`` format;
                    - ``8`` if ``encoding`` is ``"PCM_U"``, ``"ULAW"`` or ``"ALAW"``
                    - ``16`` if ``encoding`` is ``"PCM_S"`` or not provided.
                    - ``32`` if ``encoding`` is ``"PCM_F"``

    Supported formats/encodings/bit depth/compression are;

    ``"wav"``, ``"amb"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note: Default encoding/bit depth is determined by the dtype of the input Tensor.

    ``"mp3"``
        Fixed bit rate (such as 128kHz) and variable bit rate compression.
        Default: VBR with high quality.

    ``"flac"``
        - 8-bit
        - 16-bit
        - 24-bit (default)

    ``"ogg"``, ``"vorbis"``
        - Different quality level. Default: approx. 112kbps

    ``"sph"``
        - 8-bit signed integer PCM
        - 16-bit signed integer PCM
        - 24-bit signed integer PCM
        - 32-bit signed integer PCM (default)
        - 8-bit mu-law
        - 8-bit a-law
        - 16-bit a-law
        - 24-bit a-law
        - 32-bit a-law

    ``"amr-nb"``
        Bitrate ranging from 4.75 kbit/s to 12.2 kbit/s. Default: 4.75 kbit/s

    ``"gsm"``
        Lossy Speech Compression, CPU intensive.

    ``"htk"``
        Uses a default single-channel 16-bit PCM format.

    Note:
        To save into formats that ``libsox`` does not handle natively, (such as ``"mp3"``,
        ``"flac"``, ``"ogg"`` and ``"vorbis"``), your installation of ``torchaudio`` has
        to be linked to ``libsox`` and corresponding codec libraries such as ``libmad``
        or ``libmp3lame`` etc.
    