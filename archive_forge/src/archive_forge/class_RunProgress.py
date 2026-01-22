import math
import sys
import time
import param
from IPython.core.display import clear_output
from ..core.util import ProgressIndicator
class RunProgress(ProgressBar):
    """
    RunProgress breaks up the execution of a slow running command so
    that the level of completion can be displayed during execution.

    This class is designed to run commands that take a single numeric
    argument that acts additively. Namely, it is expected that a slow
    running command 'run_hook(X+Y)' can be arbitrarily broken up into
    multiple, faster executing calls 'run_hook(X)' and 'run_hook(Y)'
    without affecting the overall result.

    For instance, this is suitable for simulations where the numeric
    argument is the simulated time - typically, advancing 10 simulated
    seconds takes about twice as long as advancing by 5 seconds.
    """
    interval = param.Number(default=100, doc='\n        The run interval used to break up updates to the progress bar.')
    run_hook = param.Callable(default=param.Dynamic.time_fn.advance, doc='\n        By default updates time in param which is very fast and does\n        not need a progress bar. Should be set to some slower running\n        callable where display of progress level is desired.')

    def __init__(self, **params):
        super().__init__(**params)

    def __call__(self, value):
        """
        Execute the run_hook to a total of value, breaking up progress
        updates by the value specified by interval.
        """
        completed = 0
        while value - completed >= self.interval:
            self.run_hook(self.interval)
            completed += self.interval
            super().__call__(100 * (completed / float(value)))
        remaining = value - completed
        if remaining != 0:
            self.run_hook(remaining)
            super().__call__(100)