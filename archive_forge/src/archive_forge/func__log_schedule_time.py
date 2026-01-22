import logging
from time import time
from typing import List, Optional, Union
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.providers.backend import Backend
from qiskit.scheduler import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.utils.parallel import parallel_map
def _log_schedule_time(start_time, end_time):
    log_msg = 'Total Scheduling Time - %.5f (ms)' % ((end_time - start_time) * 1000)
    logger.info(log_msg)