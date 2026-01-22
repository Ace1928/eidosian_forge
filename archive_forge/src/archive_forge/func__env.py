import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def _env(self, config: command.PagerConfig) -> typing.Dict:
    return self.cmd.environment_variables(config) or {}