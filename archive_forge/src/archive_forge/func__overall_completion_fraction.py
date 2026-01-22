import os
import time
def _overall_completion_fraction(self, child_fraction=0.0):
    """Return fractional completion of this task and its parents

        Returns None if no completion can be computed."""
    if self.current_cnt is not None and self.total_cnt:
        own_fraction = (float(self.current_cnt) + child_fraction) / self.total_cnt
    else:
        own_fraction = child_fraction
    if self._parent_task is None:
        return own_fraction
    else:
        if own_fraction is None:
            own_fraction = 0.0
        return self._parent_task._overall_completion_fraction(own_fraction)