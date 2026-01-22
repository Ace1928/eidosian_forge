import time
import warnings
from typing import Dict, Sequence, Union, Optional, TYPE_CHECKING
from cirq_ionq import ionq_exceptions, results
from cirq._doc import document
import cirq
def _check_if_unsuccessful(self):
    if self.status() in self.UNSUCCESSFUL_STATES:
        raise ionq_exceptions.IonQUnsuccessfulJobException(self.job_id(), self.status())