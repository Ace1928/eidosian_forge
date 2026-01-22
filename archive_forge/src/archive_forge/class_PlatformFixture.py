import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class PlatformFixture(fixtures.Fixture):

    def __init__(self, platform: str):
        self.platform = platform

    def _setUp(self) -> None:
        self.addCleanup(setattr, sys, 'platform', sys.platform)
        sys.platform = self.platform