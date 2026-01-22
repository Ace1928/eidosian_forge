import logging
import threading
import time
import numpy as np
def _tune(self):
    """Changes the steps per execution using the following algorithm.

        If there is more than a 10% increase in the throughput, then the last
        recorded action is repeated (i.e. if increasing the SPE caused an
        increase in throughput, it is increased again). If there is more than a
        10% decrease in the throughput, then the opposite of the last action is
        performed (i.e. if increasing the SPE decreased the throughput, then the
        SPE is decreased).
        """
    self.avg_rgsps = sum(self.rgsps) / len(self.rgsps)
    fast_threshold = (1 + self.spe_change_threshold) * self.prev_avg_rgsps
    slow_threshold = (1 - self.spe_change_threshold) * self.prev_avg_rgsps
    if self.spe_tune_last_action_add:
        repeat_action_mult = 1.5
        opposite_action_mult = 0.5
    else:
        repeat_action_mult = 0.5
        opposite_action_mult = 1.5
    spe_variable = self._steps_per_execution
    spe_limit = spe_variable.dtype.max / 1.5
    current_spe = spe_variable.numpy().item()
    if self.avg_rgsps > fast_threshold:
        new_spe = current_spe * repeat_action_mult
    elif self.avg_rgsps < slow_threshold:
        new_spe = current_spe * opposite_action_mult
        self.spe_tune_last_action_add = not self.spe_tune_last_action_add
    else:
        new_spe = current_spe
    if current_spe >= spe_limit:
        new_spe = current_spe
    elif current_spe == 0:
        new_spe = self.init_spe
    self._steps_per_execution.assign(np.round(new_spe))
    self.prev_avg_rgsps = self.avg_rgsps