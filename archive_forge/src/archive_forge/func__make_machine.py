from automaton import exceptions as excp
from automaton import runners
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import runtime
from taskflow.patterns import linear_flow as lf
from taskflow import states as st
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import notifier
from taskflow.utils import persistence_utils as pu
def _make_machine(self, flow, initial_state=None):
    runtime = self._make_runtime(flow, initial_state=initial_state)
    machine, memory = runtime.builder.build({})
    machine_runner = runners.FiniteRunner(machine)
    return (runtime, machine, memory, machine_runner)