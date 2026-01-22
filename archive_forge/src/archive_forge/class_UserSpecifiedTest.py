import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class UserSpecifiedTest(DefaultTest):

    def setUp(self) -> None:
        with fixtures.EnvironmentVariable('FOO_PAGER', 'less "-r" +F'):
            self.cmd = command.UserSpecifiedPager('FOO_PAGER')

    def test_env_var_priority_cmd(self) -> None:
        with fixtures.EnvironmentVariable('FOO', 'foo'):
            cmd = command.UserSpecifiedPager('FOO', 'BAR')
        self.assertEqual(['foo'], cmd.command())

    def test_env_var_fallthrough_cmd(self) -> None:
        with fixtures.EnvironmentVariable('BAR', 'bar'):
            cmd = command.UserSpecifiedPager('FOO', 'BAR')
        self.assertEqual(['bar'], cmd.command())

    def test_default_cmd(self) -> None:
        with fixtures.EnvironmentVariable('FOO'):
            with fixtures.EnvironmentVariable('BAR'):
                cmd = command.UserSpecifiedPager('FOO', 'BAR')
        self.assertEqual(command.PlatformPager().command(), cmd.command())