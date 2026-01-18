import sqlite3
from pathlib import Path
from traitlets.config.application import Application
from .application import BaseIPythonApplication
from traitlets import Bool, Int, Dict
from ..utils.io import ask_yes_no

An application for managing IPython history.

To be invoked as the `ipython history` subcommand.
