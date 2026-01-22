import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
class NoOpDecider(Decider):
    """No-op decider that says it is always ok to run & has no effect(s)."""

    def tally(self, runtime):
        """Always good to go."""
        return []

    def affect(self, runtime, nay_voters):
        """Does nothing."""