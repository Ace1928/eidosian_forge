import pkg_resources
import argparse
import logging
import sys
from warnings import warn
class BaseCommandMeta(type):

    @property
    def summary(cls):
        """
        This is used to populate the --help argument on the command line.

        This provides a default behavior which takes the first sentence of the
        command's docstring and uses it.
        """
        return cls.__doc__.strip().splitlines()[0].rstrip('.')