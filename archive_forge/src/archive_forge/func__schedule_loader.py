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