import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class MoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cmd = command.More()

    def test_cmd(self) -> None:
        self.assertEqual(['more'], self.cmd.command())

    def test_less_env_defaults(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
        self.assertIsNone(self.cmd.environment_variables(config))