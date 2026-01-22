from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
def _figure_formats_changed(self, name, old, new):
    if 'jpg' in new or 'jpeg' in new:
        if not pil_available():
            raise TraitError('Requires PIL/Pillow for JPG figures')
    self._update_figure_formatters()