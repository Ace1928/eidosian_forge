import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
Compute energy histogram via ray tracing.

    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    ``num_rays`` rays are casted uniformly in all directions from the source;
    when a ray intersects a wall, it is reflected and part of its energy is absorbed.
    It is also scattered (sent directly to the microphone(s)) according to the ``scattering``
    coefficient.
    When a ray is close to the microphone, its current energy is recorded in the output
    histogram for that given time slot.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        absorption (float or torch.Tensor, optional): The absorption coefficients of wall materials.
            (Default: ``0.0``).
            If the type is ``float``, the absorption coefficient is identical to all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, representing absorption
            coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and
            ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be  `(num_bands, 6)`.
            ``num_bands`` is the number of frequency bands (usually 7).
        scattering(float or torch.Tensor, optional): The scattering coefficients of wall materials. (Default: ``0.0``)
            The shape and type of this parameter is the same as for ``absorption``.
        mic_radius(float, optional): The radius of the microphone in meters. (Default: 0.5)
        sound_speed (float, optional): The speed of sound in meters per second. (Default: ``343.0``)
        energy_thres (float, optional): The energy level below which we stop tracing a ray. (Default: ``1e-7``)
            The initial energy of each ray is ``2 / num_rays``.
        time_thres (float, optional): The maximal duration for which rays are traced. (Unit: seconds) (Default: 10.0)
        hist_bin_size (float, optional): The size of each bin in the output histogram. (Unit: seconds) (Default: 0.004)

    Returns:
        (torch.Tensor): The 3D histogram(s) where the energy of the traced ray is recorded.
            Each bin corresponds to a given time slot.
            The shape is `(channel, num_bands, num_bins)`, where
            ``num_bins = ceil(time_thres / hist_bin_size)``.
            If both ``absorption`` and ``scattering`` are floats, then ``num_bands == 1``.
    