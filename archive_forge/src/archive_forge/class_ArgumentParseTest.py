import argparse
import fixtures  # type: ignore
import typing
import autopage
import autopage.argparse
class ArgumentParseTest(fixtures.TestWithFixtures):

    def setUp(self) -> None:
        patch_ap = self.useFixture(fixtures.MockPatch('autopage.AutoPager'))
        self.pager = patch_ap.mock
        self.stream = self.useFixture(fixtures.StringStream('stdout')).stream
        self.pager.return_value.__enter__.return_value = self.stream

    def test_argparse(self, module: typing.Any=autopage.argparse, color: bool=True) -> None:
        self.pager.return_value.to_terminal.return_value = color
        parser = module.ArgumentParser()
        try:
            parser.parse_args(['foo', '--help'])
        except SystemExit as exit:
            self.assertIs(self.pager.return_value.exit_code.return_value, exit.code)
        self.pager.assert_called_once_with(None, allow_color=True, line_buffering=False, reset_on_exit=False)
        self.pager.return_value.__enter__.assert_called_once()
        self.stream.seek(0)
        self.assertEqual('\x1b' in self.stream.read(), color)

    def test_argparse_no_color(self) -> None:
        self.test_argparse(color=False)

    def test_monkey_patch(self, color: bool=True) -> None:
        patch = self.useFixture(fixtures.MockPatch('argparse._HelpAction'))
        autopage.argparse.monkey_patch()
        self.assertIsNot(patch.mock, argparse._HelpAction)
        self.test_argparse(argparse, color)

    def test_monkey_patch_no_color(self) -> None:
        self.test_monkey_patch(color=False)

    def test_monkey_patch_context(self, color: bool=True) -> None:
        patch = self.useFixture(fixtures.MockPatch('argparse._HelpAction'))
        with autopage.argparse.monkey_patch():
            self.assertIsNot(patch.mock, argparse._HelpAction)
            self.test_argparse(argparse, color)
        self.assertIs(patch.mock, argparse._HelpAction)

    def test_monkey_patch_context_no_color(self) -> None:
        self.test_monkey_patch_context(color=False)