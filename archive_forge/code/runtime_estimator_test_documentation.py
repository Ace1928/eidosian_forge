import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
Regression test

    Make sure that high numbers of qubits do not
    slow the rep rate down to below zero.
    