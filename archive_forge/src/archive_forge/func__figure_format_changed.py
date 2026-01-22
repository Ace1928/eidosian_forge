from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
def _figure_format_changed(self, name, old, new):
    if new:
        self.figure_formats = {new}