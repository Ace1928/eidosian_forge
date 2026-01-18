import os
from typing import List, Optional, Tuple
import torch
import torchaudio
from torchaudio._internal.module_utils import deprecated
from torchaudio.utils.sox_utils import list_effects
Apply sox effects to the audio file and load the resulting data as Tensor

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        This function works in the way very similar to ``sox`` command, however there are slight
        differences. For example, ``sox`` commnad adds certain effects automatically (such as
        ``rate`` effect after ``speed``, ``pitch`` etc), but this function only applies the given
        effects. Therefore, to actually apply ``speed`` effect, you also need to give ``rate``
        effect with desired sampling rate, because internally, ``speed`` effects only alter sampling
        rate and leave samples untouched.

    Args:
        path (path-like object):
            Source of audio data.
        effects (List[List[str]]): List of effects.
        normalize (bool, optional):
            When ``True``, this function converts the native sample type to ``float32``.
            Default: ``True``.

            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type.
            This argument has no effect for formats other than integer WAV type.

        channels_first (bool, optional): When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension,

    Returns:
        (Tensor, int): Resulting Tensor and sample rate.
        If ``normalize=True``, the resulting Tensor is always ``float32`` type.
        If ``normalize=False`` and the input audio file is of integer WAV file, then the
        resulting Tensor has corresponding integer type. (Note 24 bit integer type is not supported)
        If ``channels_first=True``, the resulting Tensor has dimension `[channel, time]`,
        otherwise `[time, channel]`.

    Example - Basic usage
        >>>
        >>> # Defines the effects to apply
        >>> effects = [
        ...     ['gain', '-n'],  # normalises to 0dB
        ...     ['pitch', '5'],  # 5 cent pitch shift
        ...     ['rate', '8000'],  # resample to 8000 Hz
        ... ]
        >>>
        >>> # Apply effects and load data with channels_first=True
        >>> waveform, sample_rate = apply_effects_file("data.wav", effects, channels_first=True)
        >>>
        >>> # Check the result
        >>> waveform.shape
        torch.Size([2, 8000])
        >>> waveform
        tensor([[ 5.1151e-03,  1.8073e-02,  2.2188e-02,  ...,  1.0431e-07,
                 -1.4761e-07,  1.8114e-07],
                [-2.6924e-03,  2.1860e-03,  1.0650e-02,  ...,  6.4122e-07,
                 -5.6159e-07,  4.8103e-07]])
        >>> sample_rate
        8000

    Example - Apply random speed perturbation to dataset
        >>>
        >>> # Load data from file, apply random speed perturbation
        >>> class RandomPerturbationFile(torch.utils.data.Dataset):
        ...     """Given flist, apply random speed perturbation
        ...
        ...     Suppose all the input files are at least one second long.
        ...     """
        ...     def __init__(self, flist: List[str], sample_rate: int):
        ...         super().__init__()
        ...         self.flist = flist
        ...         self.sample_rate = sample_rate
        ...
        ...     def __getitem__(self, index):
        ...         speed = 0.5 + 1.5 * random.randn()
        ...         effects = [
        ...             ['gain', '-n', '-10'],  # apply 10 db attenuation
        ...             ['remix', '-'],  # merge all the channels
        ...             ['speed', f'{speed:.5f}'],  # duration is now 0.5 ~ 2.0 seconds.
        ...             ['rate', f'{self.sample_rate}'],
        ...             ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
        ...             ['trim', '0', '2'],  # get the first 2 seconds
        ...         ]
        ...         waveform, _ = torchaudio.sox_effects.apply_effects_file(
        ...             self.flist[index], effects)
        ...         return waveform
        ...
        ...     def __len__(self):
        ...         return len(self.flist)
        ...
        >>> dataset = RandomPerturbationFile(file_list, sample_rate=8000)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        >>>     pass
    