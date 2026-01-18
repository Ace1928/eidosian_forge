from typing import List, Optional, Sequence
import cirq
Compute the estimated time for running a batch of programs.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run_batch() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        programs: a sequence of circuits to be executed
        params_list: a parameter sweep for each circuit
        repetitions: number of repetitions to execute per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    