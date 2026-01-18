import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP
def zero_vjp(_):
    res = tuple((np.zeros(mp.shape(None, tape.shots)) for mp in tape.measurements))
    return res[0] if len(tape.measurements) == 1 else res