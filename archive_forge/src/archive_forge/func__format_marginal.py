from typing import Sequence, Union, Optional, Dict, List
from collections import Counter
from copy import deepcopy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.result.result import Result
from qiskit.result.counts import Counts
from qiskit.result.distributions.probability import ProbDistribution
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.result.postprocess import _bin_to_hex
from qiskit._accelerate import results as results_rs  # pylint: disable=no-name-in-module
def _format_marginal(counts, marg_counts, indices):
    """Take the output of marginalize and add placeholders for
    multiple cregs and non-indices."""
    format_counts = {}
    counts_template = next(iter(counts))
    counts_len = len(counts_template.replace(' ', ''))
    indices_rev = sorted(indices, reverse=True)
    for count in marg_counts:
        index_dict = dict(zip(indices_rev, count))
        count_bits = ''.join([index_dict[index] if index in index_dict else '_' for index in range(counts_len)])[::-1]
        for index, bit in enumerate(counts_template):
            if bit == ' ':
                count_bits = count_bits[:index] + ' ' + count_bits[index:]
        format_counts[count_bits] = marg_counts[count]
    return format_counts