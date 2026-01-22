import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats:
    """Top level container for Callgrind results collected by Timer.

    Manipulation is generally done using the FunctionCounts class, which is
    obtained by calling `CallgrindStats.stats(...)`. Several convenience
    methods are provided as well; the most significant is
    `CallgrindStats.as_standardized()`.
    """
    task_spec: common.TaskSpec
    number_per_run: int
    built_with_debug_symbols: bool
    baseline_inclusive_stats: FunctionCounts
    baseline_exclusive_stats: FunctionCounts
    stmt_inclusive_stats: FunctionCounts
    stmt_exclusive_stats: FunctionCounts
    stmt_callgrind_out: Optional[str]

    def __repr__(self) -> str:
        newline = '\n'
        base_stats = self.baseline_exclusive_stats
        output = f'\n{super().__repr__()}\n{self.task_spec.summarize()}\n  {'':>25}All{'':>10}Noisy symbols removed\n    Instructions: {self.counts(denoise=False):>12}{'':>15}{self.counts(denoise=True):>12}\n    Baseline:     {base_stats.sum():>12}{'':>15}{base_stats.denoise().sum():>12}\n{self.number_per_run} runs per measurement, {self.task_spec.num_threads} thread{('s' if self.task_spec.num_threads > 1 else '')}\n'.strip()
        if not self.built_with_debug_symbols:
            output += textwrap.dedent('\n            Warning: PyTorch was not built with debug symbols.\n                     Source information may be limited. Rebuild with\n                     REL_WITH_DEB_INFO=1 for more detailed results.')
        return output

    def stats(self, inclusive: bool=False) -> FunctionCounts:
        """Returns detailed function counts.

        Conceptually, the FunctionCounts returned can be thought of as a tuple
        of (count, path_and_function_name) tuples.

        `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        reducing noise when diffing counts from two different runs. (See
        CallgrindStats.delta(...) for more details)
        """
        return self.stmt_inclusive_stats if inclusive else self.stmt_exclusive_stats

    def counts(self, *, denoise: bool=False) -> int:
        """Returns the total number of instructions executed.

        See `FunctionCounts.denoise()` for an explanation of the `denoise` arg.
        """
        stats = self.stmt_exclusive_stats
        return (stats.denoise() if denoise else stats).sum()

    def delta(self, other: 'CallgrindStats', inclusive: bool=False) -> FunctionCounts:
        """Diff two sets of counts.

        One common reason to collect instruction counts is to determine the
        the effect that a particular change will have on the number of instructions
        needed to perform some unit of work. If a change increases that number, the
        next logical question is "why". This generally involves looking at what part
        if the code increased in instruction count. This function automates that
        process so that one can easily diff counts on both an inclusive and
        exclusive basis.
        """
        return self.stats(inclusive=inclusive) - other.stats(inclusive=inclusive)

    def as_standardized(self) -> 'CallgrindStats':
        """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, on stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivalent call sites
        when diffing.
        """

        def strip(stats: FunctionCounts) -> FunctionCounts:
            transforms = (('^.+build/\\.\\./', 'build/../'), ('^.+/' + re.escape('build/aten/'), 'build/aten/'), ('^.+/' + re.escape('Python/'), 'Python/'), ('^.+/' + re.escape('Objects/'), 'Objects/'), ('\\s\\[.+\\]$', ''))
            for before, after in transforms:
                stats = stats.transform(lambda fn: re.sub(before, after, fn))
            return stats
        return CallgrindStats(task_spec=self.task_spec, number_per_run=self.number_per_run, built_with_debug_symbols=self.built_with_debug_symbols, baseline_inclusive_stats=strip(self.baseline_inclusive_stats), baseline_exclusive_stats=strip(self.baseline_exclusive_stats), stmt_inclusive_stats=strip(self.stmt_inclusive_stats), stmt_exclusive_stats=strip(self.stmt_exclusive_stats), stmt_callgrind_out=None)