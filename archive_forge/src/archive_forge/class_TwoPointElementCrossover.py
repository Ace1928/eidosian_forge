import numpy as np
from ase.ga.offspring_creator import OffspringCreator
class TwoPointElementCrossover(ElementCrossover):
    """Crosses two individuals by choosing two cross points
    at random"""

    def __init__(self, element_pool, max_diff_elements=None, min_percentage_elements=None, verbose=False):
        ElementCrossover.__init__(self, element_pool, max_diff_elements, min_percentage_elements, verbose)
        self.descriptor = 'TwoPointElementCrossover'

    def get_new_individual(self, parents):
        raise NotImplementedError