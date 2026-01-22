import argparse
import io
import os
import fixtures
import testtools
from openstackclient.tests.unit import fakes
class CompareBySet(list):
    """Class to compare value using set."""

    def __eq__(self, other):
        return set(self) == set(other)