import datetime
from typing import Dict, List, Optional, Sequence, Set, Union
import cirq
from cirq_google.engine.abstract_job import AbstractJob
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_engine import AbstractEngine
from cirq_google.cloud import quantum
Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.

        Raises:
            ValueError: if multiple processor ids are given.
        