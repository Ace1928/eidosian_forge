import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
Create a mfcc from a raw audio signal. This matches the input/output of Kaldi's
    compute-mfcc-feats.

    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        cepstral_lifter (float, optional): Constant that controls scaling of MFCCs (Default: ``22.0``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float, optional): High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
         (Default: ``0.0``)
        htk_compat (bool, optional): If true, put energy last.  Warning: not sufficient to get HTK compatible
         features (need to change other parameters). (Default: ``False``)
        low_freq (float, optional): Low cutoff frequency for mel bins (Default: ``20.0``)
        num_ceps (int, optional): Number of cepstra in MFCC computation (including C0) (Default: ``13``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int, optional): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool, optional): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float, optional): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``"povey"``)

    Returns:
        Tensor: A mfcc identical to what Kaldi would output. The shape is (m, ``num_ceps``)
        where m is calculated in _get_strided
    