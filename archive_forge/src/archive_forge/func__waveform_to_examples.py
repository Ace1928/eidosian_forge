import math
import torch
def _waveform_to_examples(data):
    log_mel = _log_mel_spectrogram(data, audio_sample_rate=_SAMPLE_RATE, log_offset=_LOG_OFFSET, window_length_secs=_STFT_WINDOW_LENGTH_SECONDS, hop_length_secs=_STFT_HOP_LENGTH_SECONDS, num_mel_bins=_NUM_BANDS, lower_edge_hertz=_MEL_MIN_HZ, upper_edge_hertz=_MEL_MAX_HZ)
    features_sample_rate = 1.0 / _STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(_EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(_EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = _frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)
    return log_mel_examples.unsqueeze(1)