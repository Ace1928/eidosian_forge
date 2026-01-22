from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
class LoConfig:
    """Pulse channel LO frequency container."""

    def __init__(self, channel_los: dict[DriveChannel | MeasureChannel, float] | None=None, lo_ranges: dict[DriveChannel | MeasureChannel, LoRange | tuple[int, int]] | None=None):
        """Lo channel configuration data structure.

        Args:
            channel_los: Dictionary of mappings from configurable channel to lo
            lo_ranges: Dictionary of mappings to be enforced from configurable channel to `LoRange`

        Raises:
            PulseError: If channel is not configurable or set lo is out of range.

        """
        self._q_lo_freq: dict[DriveChannel, float] = {}
        self._m_lo_freq: dict[MeasureChannel, float] = {}
        self._lo_ranges: dict[DriveChannel | MeasureChannel, LoRange] = {}
        lo_ranges = lo_ranges if lo_ranges else {}
        for channel, freq in lo_ranges.items():
            self.add_lo_range(channel, freq)
        channel_los = channel_los if channel_los else {}
        for channel, freq in channel_los.items():
            self.add_lo(channel, freq)

    def add_lo(self, channel: DriveChannel | MeasureChannel, freq: float):
        """Add a lo mapping for a channel."""
        if isinstance(channel, DriveChannel):
            self.check_lo(channel, freq)
            self._q_lo_freq[channel] = freq
        elif isinstance(channel, MeasureChannel):
            self.check_lo(channel, freq)
            self._m_lo_freq[channel] = freq
        else:
            raise PulseError('Specified channel %s cannot be configured.' % channel.name)

    def add_lo_range(self, channel: DriveChannel | MeasureChannel, lo_range: LoRange | tuple[int, int]):
        """Add lo range to configuration.

        Args:
            channel: Channel to add lo range for
            lo_range: Lo range to add

        """
        if isinstance(lo_range, (list, tuple)):
            lo_range = LoRange(*lo_range)
        self._lo_ranges[channel] = lo_range

    def check_lo(self, channel: DriveChannel | MeasureChannel, freq: float) -> bool:
        """Check that lo is valid for channel.

        Args:
            channel: Channel to validate lo for
            freq: lo frequency
        Raises:
            PulseError: If freq is outside of channels range
        Returns:
            True if lo is valid for channel
        """
        lo_ranges = self._lo_ranges
        if channel in lo_ranges:
            lo_range = lo_ranges[channel]
            if not lo_range.includes(freq):
                raise PulseError(f'Specified LO freq {freq:f} is out of range {lo_range}')
        return True

    def channel_lo(self, channel: DriveChannel | MeasureChannel) -> float:
        """Return channel lo.

        Args:
            channel: Channel to get lo for
        Raises:
            PulseError: If channel is not configured
        Returns:
            Lo of supplied channel if present
        """
        if isinstance(channel, DriveChannel):
            if channel in self.qubit_los:
                return self.qubit_los[channel]
        if isinstance(channel, MeasureChannel):
            if channel in self.meas_los:
                return self.meas_los[channel]
        raise PulseError('Channel %s is not configured' % channel)

    @property
    def qubit_los(self) -> dict[DriveChannel, float]:
        """Returns dictionary mapping qubit channels (DriveChannel) to los."""
        return self._q_lo_freq

    @property
    def meas_los(self) -> dict[MeasureChannel, float]:
        """Returns dictionary mapping measure channels (MeasureChannel) to los."""
        return self._m_lo_freq