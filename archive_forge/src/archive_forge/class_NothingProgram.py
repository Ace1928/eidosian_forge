import datetime
import pytest
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
class NothingProgram(AbstractLocalProgram):

    def delete(self, delete_jobs: bool=False) -> None:
        pass

    def delete_job(self, job_id: str) -> None:
        pass