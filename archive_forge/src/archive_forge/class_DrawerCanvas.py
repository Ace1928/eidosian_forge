from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
class DrawerCanvas:
    """Collection of `Chart` and configuration data.

    Pulse channels are associated with some `Chart` instance and
    drawing data object are stored in the `Chart` instance.

    Device, stylesheet, and some user generators are stored in the `DrawingCanvas`
    and `Chart` instances are also attached to the `DrawerCanvas` as children.
    Global configurations are accessed by those children to modify
    the appearance of the `Chart` output.
    """

    def __init__(self, stylesheet: QiskitPulseStyle, device: device_info.DrawerBackendInfo):
        """Create new data container with backend system information.

        Args:
            stylesheet: Stylesheet to decide appearance of output image.
            device: Backend information to run the program.
        """
        self.formatter = stylesheet.formatter
        self.generator = stylesheet.generator
        self.layout = stylesheet.layout
        self.device = device
        self.global_charts = Chart(parent=self, name='global')
        self.charts: list[Chart] = []
        self.disable_chans: set[pulse.channels.Channel] = set()
        self.disable_types: set[str] = set()
        self.chan_scales: dict[pulse.channels.DriveChannel | pulse.channels.MeasureChannel | pulse.channels.ControlChannel | pulse.channels.AcquireChannel, float] = {}
        self._time_range = (0, 0)
        self._time_breaks: list[tuple[int, int]] = []
        self.fig_title = ''

    @property
    def time_range(self) -> tuple[int, int]:
        """Return current time range to draw.

        Calculate net duration and add side margin to edge location.

        Returns:
            Time window considering side margin.
        """
        t0, t1 = self._time_range
        total_time_elimination = 0
        for t0b, t1b in self.time_breaks:
            if t1b > t0 and t0b < t1:
                total_time_elimination += t1b - t0b
        net_duration = t1 - t0 - total_time_elimination
        new_t0 = t0 - net_duration * self.formatter['margin.left_percent']
        new_t1 = t1 + net_duration * self.formatter['margin.right_percent']
        return (new_t0, new_t1)

    @time_range.setter
    def time_range(self, new_range: tuple[int, int]):
        """Update time range to draw."""
        self._time_range = new_range

    @property
    def time_breaks(self) -> list[tuple[int, int]]:
        """Return time breaks with time range.

        If an edge of time range is in the axis break period,
        the axis break period is recalculated.

        Raises:
            VisualizationError: When axis break is greater than time window.

        Returns:
            List of axis break periods considering the time window edges.
        """
        t0, t1 = self._time_range
        axis_breaks = []
        for t0b, t1b in self._time_breaks:
            if t0b >= t1 or t1b <= t0:
                continue
            if t0b < t0 and t1b > t1:
                raise VisualizationError('Axis break is greater than time window. Nothing will be drawn.')
            if t0b < t0 < t1b:
                if t1b - t0 > self.formatter['axis_break.length']:
                    new_t0 = t0 + 0.5 * self.formatter['axis_break.max_length']
                    axis_breaks.append((new_t0, t1b))
                continue
            if t0b < t1 < t1b:
                if t1 - t0b > self.formatter['axis_break.length']:
                    new_t1 = t1 - 0.5 * self.formatter['axis_break.max_length']
                    axis_breaks.append((t0b, new_t1))
                continue
            axis_breaks.append((t0b, t1b))
        return axis_breaks

    @time_breaks.setter
    def time_breaks(self, new_breaks: list[tuple[int, int]]):
        """Set new time breaks."""
        self._time_breaks = sorted(new_breaks, key=lambda x: x[0])

    def load_program(self, program: pulse.Waveform | pulse.SymbolicPulse | pulse.Schedule | pulse.ScheduleBlock):
        """Load a program to draw.

        Args:
            program: Pulse program or waveform to draw.

        Raises:
            VisualizationError: When input program is invalid data format.
        """
        if isinstance(program, (pulse.Schedule, pulse.ScheduleBlock)):
            self._schedule_loader(program)
        elif isinstance(program, (pulse.Waveform, pulse.SymbolicPulse)):
            self._waveform_loader(program)
        else:
            raise VisualizationError('Data type %s is not supported.' % type(program))
        self.set_time_range(0, program.duration, seconds=False)
        self.fig_title = self.layout['figure_title'](program=program, device=self.device)

    def _waveform_loader(self, program: pulse.Waveform | pulse.SymbolicPulse):
        """Load Waveform instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Waveform` to draw.
        """
        chart = Chart(parent=self)
        fake_inst = pulse.Play(program, types.WaveformChannel())
        inst_data = types.PulseInstruction(t0=0, dt=self.device.dt, frame=types.PhaseFreqTuple(phase=0, freq=0), inst=fake_inst, is_opaque=program.is_parameterized())
        for gen in self.generator['waveform']:
            obj_generator = partial(gen, formatter=self.formatter, device=self.device)
            for data in obj_generator(inst_data):
                chart.add_data(data)
        self.charts.append(chart)

    def _schedule_loader(self, program: pulse.Schedule | pulse.ScheduleBlock):
        """Load Schedule instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Schedule` to draw.
        """
        program = target_qobj_transform(program, remove_directives=False)
        self.chan_scales = {}
        for chan in program.channels:
            if isinstance(chan, pulse.channels.DriveChannel):
                self.chan_scales[chan] = self.formatter['channel_scaling.drive']
            elif isinstance(chan, pulse.channels.MeasureChannel):
                self.chan_scales[chan] = self.formatter['channel_scaling.measure']
            elif isinstance(chan, pulse.channels.ControlChannel):
                self.chan_scales[chan] = self.formatter['channel_scaling.control']
            elif isinstance(chan, pulse.channels.AcquireChannel):
                self.chan_scales[chan] = self.formatter['channel_scaling.acquire']
            else:
                self.chan_scales[chan] = 1.0
        mapper = self.layout['chart_channel_map']
        for name, chans in mapper(channels=program.channels, formatter=self.formatter, device=self.device):
            chart = Chart(parent=self, name=name)
            for chan in chans:
                chart.load_program(program=program, chan=chan)
            barrier_sched = program.filter(instruction_types=[pulse.instructions.RelativeBarrier], channels=chans)
            for t0, _ in barrier_sched.instructions:
                inst_data = types.BarrierInstruction(t0, self.device.dt, chans)
                for gen in self.generator['barrier']:
                    obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                    for data in obj_generator(inst_data):
                        chart.add_data(data)
            chart_axis = types.ChartAxis(name=chart.name, channels=chart.channels)
            for gen in self.generator['chart']:
                obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                for data in obj_generator(chart_axis):
                    chart.add_data(data)
            self.charts.append(chart)
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.Snapshot])
        for t0, inst in snapshot_sched.instructions:
            inst_data = types.SnapshotInstruction(t0, self.device.dt, inst.label, inst.channels)
            for gen in self.generator['snapshot']:
                obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                for data in obj_generator(inst_data):
                    self.global_charts.add_data(data)
        self.time_breaks = self._calculate_axis_break(program)

    def _calculate_axis_break(self, program: pulse.Schedule) -> list[tuple[int, int]]:
        """A helper function to calculate axis break of long pulse sequence.

        Args:
            program: A schedule to calculate axis break.

        Returns:
            List of axis break periods.
        """
        axis_breaks = []
        edges = set()
        for t0, t1 in chain.from_iterable(program.timeslots.values()):
            if t1 - t0 > 0:
                edges.add(t0)
                edges.add(t1)
        edges = sorted(edges)
        for t0, t1 in zip(edges[:-1], edges[1:]):
            if t1 - t0 > self.formatter['axis_break.length']:
                t_l = t0 + 0.5 * self.formatter['axis_break.max_length']
                t_r = t1 - 0.5 * self.formatter['axis_break.max_length']
                axis_breaks.append((t_l, t_r))
        return axis_breaks

    def set_time_range(self, t_start: float, t_end: float, seconds: bool=True):
        """Set time range to draw.

        All child chart instances are updated when time range is updated.

        Args:
            t_start: Left boundary of drawing in units of cycle time or real time.
            t_end: Right boundary of drawing in units of cycle time or real time.
            seconds: Set `True` if times are given in SI unit rather than dt.

        Raises:
            VisualizationError: When times are given in float without specifying dt.
        """
        if seconds:
            if self.device.dt is not None:
                t_start = int(np.round(t_start / self.device.dt))
                t_end = int(np.round(t_end / self.device.dt))
            else:
                raise VisualizationError('Setting time range with SI units requires backend `dt` information.')
        self.time_range = (t_start, t_end)

    def set_disable_channel(self, channel: pulse.channels.Channel, remove: bool=True):
        """Interface method to control visibility of pulse channels.

        Specified object in the blocked list will not be shown.

        Args:
            channel: A pulse channel object to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if remove:
            self.disable_chans.add(channel)
        else:
            self.disable_chans.discard(channel)

    def set_disable_type(self, data_type: types.DataTypes, remove: bool=True):
        """Interface method to control visibility of data types.

        Specified object in the blocked list will not be shown.

        Args:
            data_type: A drawing data type to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if isinstance(data_type, Enum):
            data_type_str = str(data_type.value)
        else:
            data_type_str = data_type
        if remove:
            self.disable_types.add(data_type_str)
        else:
            self.disable_types.discard(data_type_str)

    def update(self):
        """Update all associated charts and generate actual drawing data from template object.

        This method should be called before the canvas is passed to the plotter.
        """
        for chart in self.charts:
            chart.update()