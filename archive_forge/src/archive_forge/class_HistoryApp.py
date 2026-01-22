import sqlite3
from pathlib import Path
from traitlets.config.application import Application
from .application import BaseIPythonApplication
from traitlets import Bool, Int, Dict
from ..utils.io import ask_yes_no
class HistoryApp(Application):
    name = 'ipython-history'
    description = 'Manage the IPython history database.'
    subcommands = Dict(dict(trim=(HistoryTrim, HistoryTrim.description.splitlines()[0]), clear=(HistoryClear, HistoryClear.description.splitlines()[0])))

    def start(self):
        if self.subapp is None:
            print('No subcommand specified. Must specify one of: %s' % self.subcommands.keys())
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()