from __future__ import annotations
import logging # isort:skip
from argparse import Namespace
from bokeh import sampledata
from ..subcommand import Subcommand
class Sampledata(Subcommand):
    """ Subcommand to download bokeh sample data sets.

    """
    name = 'sampledata'
    help = 'Download the bokeh sample data sets'

    def invoke(self, args: Namespace) -> None:
        """

        """
        sampledata.download()