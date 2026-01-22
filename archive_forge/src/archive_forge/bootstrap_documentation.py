from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from typing import Sequence
from bokeh import __version__
from bokeh.settings import settings
from bokeh.util.strings import nice_join
from . import subcommands
from .util import die
 Execute the Bokeh command.

    Args:
        argv (seq[str]) : a list of command line arguments to process

    Returns:
        None

    The first item in ``argv`` is typically "bokeh", and the second should
    be the name of one of the available subcommands:

    * :ref:`info <bokeh.command.subcommands.info>`
    * :ref:`json <bokeh.command.subcommands.json>`
    * :ref:`sampledata <bokeh.command.subcommands.sampledata>`
    * :ref:`secret <bokeh.command.subcommands.secret>`
    * :ref:`serve <bokeh.command.subcommands.serve>`
    * :ref:`static <bokeh.command.subcommands.static>`

    