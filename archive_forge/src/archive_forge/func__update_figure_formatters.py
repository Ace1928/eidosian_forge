from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
def _update_figure_formatters(self):
    if self.shell is not None:
        from IPython.core.pylabtools import select_figure_formats
        select_figure_formats(self.shell, self.figure_formats, **self.print_figure_kwargs)