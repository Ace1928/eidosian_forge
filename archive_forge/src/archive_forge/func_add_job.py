import copy
import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_program import AbstractProgram
def add_job(self, job_id: str, job: 'AbstractLocalJob') -> None:
    self._jobs[job_id] = job