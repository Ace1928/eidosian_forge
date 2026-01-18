from lognormal_around import lognormal_around
import numpy as np
import random
def update_refractory_period(self, activity_level):
    """
        Update the refractory period based on neuron activity level.
        """
    self.refractory_period = int(lognormal_around(5.5 + activity_level, 1, 10) * self.global_scaling_factor)