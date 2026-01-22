import random
import numpy as np
from functools import partial
from operator import attrgetter
def _fitTournament(individuals, k, select):
    chosen = []
    for i in range(k):
        aspirants = select(individuals, k=fitness_size)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen