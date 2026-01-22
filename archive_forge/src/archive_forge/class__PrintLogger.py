import abc
from typing import TYPE_CHECKING
class _PrintLogger(_WorkflowLogger):

    def __init__(self, n_total: int):
        self.n_total = n_total
        self.i = 0

    def initialize(self):
        """Write a newline at the start of an execution loop."""
        print()

    def consume_result(self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Print a simple count of completed executables."""
        print(f'\r{self.i + 1} / {self.n_total}', end='', flush=True)
        self.i += 1

    def finalize(self):
        """Write a newline at the end of an execution loop."""
        print()