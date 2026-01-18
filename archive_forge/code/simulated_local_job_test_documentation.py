import pytest
import numpy as np
import sympy
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
Tests for SimulatedLocalJob.