import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.

    This class stores one or more measurements of a given statement. It is
    serializable and provides several convenience methods
    (including a detailed __repr__) for downstream consumers.
    """
    number_per_run: int
    raw_times: List[float]
    task_spec: TaskSpec
    metadata: Optional[Dict[Any, Any]] = None

    def __post_init__(self) -> None:
        self._sorted_times: Tuple[float, ...] = ()
        self._warnings: Tuple[str, ...] = ()
        self._median: float = -1.0
        self._mean: float = -1.0
        self._p25: float = -1.0
        self._p75: float = -1.0

    def __getattr__(self, name: str) -> Any:
        if name in _TASKSPEC_FIELDS:
            return getattr(self.task_spec, name)
        return super().__getattribute__(name)

    @property
    def times(self) -> List[float]:
        return [t / self.number_per_run for t in self.raw_times]

    @property
    def median(self) -> float:
        self._lazy_init()
        return self._median

    @property
    def mean(self) -> float:
        self._lazy_init()
        return self._mean

    @property
    def iqr(self) -> float:
        self._lazy_init()
        return self._p75 - self._p25

    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation used in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
        self._lazy_init()
        n_total = len(self._sorted_times)
        lower_bound = int(n_total // 4)
        upper_bound = int(torch.tensor(3 * n_total / 4).ceil())
        interquartile_points: Tuple[float, ...] = self._sorted_times[lower_bound:upper_bound]
        std = torch.tensor(interquartile_points).std(unbiased=False).item()
        sqrt_n = torch.tensor(len(interquartile_points)).sqrt().item()
        confidence_interval = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = torch.tensor(self._median / confidence_interval).log10().item()
        num_significant_figures = int(torch.tensor(relative_ci).floor())
        return min(max(num_significant_figures, 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def has_warnings(self) -> bool:
        self._lazy_init()
        return bool(self._warnings)

    def _lazy_init(self) -> None:
        if self.raw_times and (not self._sorted_times):
            self._sorted_times = tuple(sorted(self.times))
            _sorted_times = torch.tensor(self._sorted_times, dtype=torch.float64)
            self._median = _sorted_times.quantile(0.5).item()
            self._mean = _sorted_times.mean().item()
            self._p25 = _sorted_times.quantile(0.25).item()
            self._p75 = _sorted_times.quantile(0.75).item()

            def add_warning(msg: str) -> None:
                rel_iqr = self.iqr / self.median * 100
                self._warnings += (f'  WARNING: Interquartile range is {rel_iqr:.1f}% of the median measurement.\n           {msg}',)
            if not self.meets_confidence(_IQR_GROSS_WARN_THRESHOLD):
                add_warning('This suggests significant environmental influence.')
            elif not self.meets_confidence(_IQR_WARN_THRESHOLD):
                add_warning('This could indicate system fluctuation.')

    def meets_confidence(self, threshold: float=_IQR_WARN_THRESHOLD) -> bool:
        return self.iqr / self.median < threshold

    @property
    def title(self) -> str:
        return self.task_spec.title

    @property
    def env(self) -> str:
        return 'Unspecified env' if self.taskspec.env is None else cast(str, self.taskspec.env)

    @property
    def as_row_name(self) -> str:
        return self.sub_label or self.stmt or '[Unknown]'

    def __repr__(self) -> str:
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
        self._lazy_init()
        skip_line, newline = ('MEASUREMENT_REPR_SKIP_LINE', '\n')
        n = len(self._sorted_times)
        time_unit, time_scale = select_unit(self._median)
        iqr_filter = '' if n >= 4 else skip_line
        repr_str = f'\n{super().__repr__()}\n{self.task_spec.summarize()}\n  {('Median: ' if n > 1 else '')}{self._median / time_scale:.2f} {time_unit}\n  {iqr_filter}IQR:    {self.iqr / time_scale:.2f} {time_unit} ({self._p25 / time_scale:.2f} to {self._p75 / time_scale:.2f})\n  {n} measurement{('s' if n > 1 else '')}, {self.number_per_run} runs {('per measurement,' if n > 1 else ',')} {self.num_threads} thread{('s' if self.num_threads > 1 else '')}\n{newline.join(self._warnings)}'.strip()
        return '\n'.join((l for l in repr_str.splitlines(keepends=False) if skip_line not in l))

    @staticmethod
    def merge(measurements: Iterable['Measurement']) -> List['Measurement']:
        """Convenience method for merging replicates.

        Merge will extrapolate times to `number_per_run=1` and will not
        transfer any metadata. (Since it might differ between replicates)
        """
        grouped_measurements: DefaultDict[TaskSpec, List[Measurement]] = collections.defaultdict(list)
        for m in measurements:
            grouped_measurements[m.task_spec].append(m)

        def merge_group(task_spec: TaskSpec, group: List['Measurement']) -> 'Measurement':
            times: List[float] = []
            for m in group:
                times.extend(m.times)
            return Measurement(number_per_run=1, raw_times=times, task_spec=task_spec, metadata=None)
        return [merge_group(t, g) for t, g in grouped_measurements.items()]