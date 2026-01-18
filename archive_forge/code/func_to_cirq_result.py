import collections
from typing import Dict, Counter, List, Optional, Sequence
import numpy as np
import cirq
def to_cirq_result(self, params: Optional[cirq.ParamResolver]=None, seed: cirq.RANDOM_STATE_OR_SEED_LIKE=None, override_repetitions=None) -> cirq.Result:
    """Samples from the simulation probability result, producing a `cirq.Result`.

        The IonQ simulator returns the probabilities of different bitstrings. This converts such
        a representation to a randomly generated sample from the simulator. Note that it does this
        on every subsequent call of this method, so repeated calls do not produce the same
        `cirq.Result`s. When a job was created by the IonQ API, it had a number of repetitions and
        this is used, unless `override_repetitions` is set here.

        Args:
            params: Any parameters which were used to generated this result.
            seed: What to use for generating the randomness. If None, then `np.random` is used.
                If an integer, `np.random.RandomState(seed) is used. Otherwise if another
                randomness generator is used, it will be used.
            override_repetitions: Repetitions were supplied when the IonQ API ran the simulation,
                but different repetitions can be supplied here and will override.

        Returns:
            A `cirq.Result` corresponding to a sample from the probability distribution returned
            from the simulator.

        Raises:
            ValueError: If the circuit used to produce this result had no measurement gates
                (and hence no measurement keys).
        """
    if len(self.measurement_dict()) == 0:
        raise ValueError('Can convert to cirq results only if the circuit had measurement gates with measurement keys.')
    rand = cirq.value.parse_random_state(seed)
    measurements = {}
    values, weights = zip(*list(self.probabilities().items()))
    indices = rand.choice(range(len(values)), p=weights, size=override_repetitions or self.repetitions())
    rand_values = np.array(values)[indices]
    for key, targets in self.measurement_dict().items():
        bits = [[value >> self.num_qubits() - target - 1 & 1 for target in targets] for value in rand_values]
        measurements[key] = np.array(bits)
    return cirq.ResultDict(params=params or cirq.ParamResolver({}), measurements=measurements)