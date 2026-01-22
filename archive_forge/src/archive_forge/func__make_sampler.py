from unittest.mock import patch
import copy
import numpy as np
import sympy
import pytest
import cirq
import cirq_pasqal
def _make_sampler(device) -> cirq_pasqal.PasqalSampler:
    sampler = cirq_pasqal.PasqalSampler(remote_host='http://00.00.00/', access_token='N/A', device=device)
    return sampler