import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class EventList(list):
    """A list of Events (for pretty printing)."""

    def __init__(self, *args, **kwargs):
        use_cuda = kwargs.pop('use_cuda', True)
        use_device = kwargs.pop('use_device', None)
        profile_memory = kwargs.pop('profile_memory', False)
        with_flops = kwargs.pop('with_flops', False)
        super().__init__(*args, **kwargs)
        self._use_cuda = use_cuda
        self._use_device = use_device
        self._profile_memory = profile_memory
        self._tree_built = False
        self._with_flops = with_flops

    def _build_tree(self):
        self._populate_cpu_children()
        self._remove_dup_nodes()
        self._set_backward_stacktraces()
        self._tree_built = True

    def __str__(self):
        return self.table()

    def _remove_dup_nodes(self):
        while True:
            to_delete = set()
            for idx in range(len(self)):
                if self[idx].cpu_parent is not None and self[idx].cpu_parent.name == self[idx].name and (len(self[idx].cpu_parent.cpu_children) == 1):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.add(idx)
            if len(to_delete) == 0:
                break
            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]
            self.clear()
            self.extend(new_evts)

    def _populate_cpu_children(self):
        """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
        sync_events = [evt for evt in self if not evt.is_async and evt.device_type == DeviceType.CPU]
        events = sorted(sync_events, key=attrgetter('thread'))
        threads = itertools.groupby(events, key=lambda event: (event.thread, event.node_id))
        for thread_id, thread_events in threads:
            thread_events_ = sorted(thread_events, key=lambda event: [event.time_range.start, -event.time_range.end])
            current_events: List[FunctionEvent] = []
            cur_end = 0
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if event.time_range.start >= parent.time_range.end or event.time_range.end > parent.time_range.end:
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        assert event.cpu_parent is None, f'There is already a CPU parent event for {event.key}'
                        event.set_cpu_parent(parent)
                        break
                current_events.append(event)

    def _set_backward_stacktraces(self):

        def bw_parent(evt):
            if evt is None:
                return None
            elif evt.scope == 1:
                return evt
            else:
                return bw_parent(evt.cpu_parent)
        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None and evt.stack is not None:
                t = (evt.sequence_nr, evt.thread)
                if t not in fwd_stacks:
                    fwd_stacks[t] = evt.stack
        for evt in self:
            p = bw_parent(evt)
            if p is not None:
                assert p.fwd_thread is not None
                t = (p.sequence_nr, p.fwd_thread)
                if t in fwd_stacks:
                    evt.stack = fwd_stacks[t]
                else:
                    evt.stack = []

    @property
    def self_cpu_time_total(self):
        return sum([event.self_cpu_time_total for event in self])

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55, max_shapes_column_width=80, header=None, top_level_events_only=False):
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
        return _build_table(self, sort_by=sort_by, row_limit=row_limit, max_src_column_width=max_src_column_width, max_name_column_width=max_name_column_width, max_shapes_column_width=max_shapes_column_width, header=header, profile_memory=self._profile_memory, with_flops=self._with_flops, top_level_events_only=top_level_events_only)

    def export_chrome_trace(self, path):
        """Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os
        device_name = 'cuda' if not self._use_device else self._use_device
        with open(path, 'w') as f:
            chrome_events = []
            next_id = 0
            f.write('[')
            for evt in self:
                if evt.trace_name is None:
                    continue
                f.write('{{"name": "{}", "ph": "X", "ts": {}, "dur": {}, "tid": {}, "pid": "CPU functions", "args": {{}}}}, '.format(evt.trace_name, evt.time_range.start, evt.time_range.elapsed_us(), evt.thread if not evt.is_remote else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "'))
                for k in evt.kernels:
                    f.write(f'{{"name": "{evt.trace_name}", "ph": "s", "ts": {evt.time_range.start}, "tid": {evt.thread}, "pid": "CPU functions", "id": {next_id}, "cat": "cpu_to_{device_name}", "args": {{}}}}, ')
                    next_id += 1
            if len(self) > 0:
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()
            f.write(']')

    def supported_export_stacks_metrics(self):
        return ['self_cpu_time_total', 'self_cuda_time_total', 'self_privateuse1_time_total']

    def export_stacks(self, path: str, metric: str):
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError('metric should be one of: ' + str(self.supported_export_stacks_metrics()))
        translate_table = str.maketrans(' ;\t\n', '____')
        with open(path, 'w') as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(evt, metric)
                    if int(metric_value) > 0:
                        stack_str = ''
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ';'
                        stack_str = stack_str[:-1] + ' ' + str(int(metric_value))
                        f.write(stack_str + '\n')

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        assert self._tree_built
        stats: Dict[Tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes, group_by_stack_n) -> Tuple[str, ...]:
            key = [str(event.key), str(event.node_id), str(event.device_type), str(event.is_legacy)]
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)
        for evt in self:
            stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt)
        avg_list = EventList(stats.values(), use_cuda=self._use_cuda, use_device=self._use_device, profile_memory=self._profile_memory, with_flops=self._with_flops)
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ''
        return avg_list

    def total_average(self):
        """Averages all events.

        Returns:
            A FunctionEventAvg object.
        """
        total_stat = FunctionEventAvg()
        for evt in self:
            total_stat += evt
            total_stat.key = None
        total_stat.key = 'Total'
        return total_stat