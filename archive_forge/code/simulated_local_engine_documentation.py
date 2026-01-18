from typing import List
from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
Collection of processors backed by local samplers.

    This class is a wrapper around `AbstractLocalEngine` and
    adds no additional functionality and exists for naming consistency
    and for possible future extension.

    This class assumes that all processors are local.  Processors
    are given during initialization.  Program and job querying
    functionality is done by serially querying all child processors.

    