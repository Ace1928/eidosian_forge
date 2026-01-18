from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
Produce output for pass_manager.