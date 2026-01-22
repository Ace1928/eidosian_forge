from copy import deepcopy
from warnings import warn
from traitlets import Bool, Unicode, default
from nbconvert.preprocessors.base import Preprocessor
from .html import HTMLExporter
@default('reveal_url_prefix')
def _reveal_url_prefix_default(self):
    if 'RevealHelpPreprocessor.url_prefix' in self.config:
        warn('Please update RevealHelpPreprocessor.url_prefix to SlidesExporter.reveal_url_prefix in config files.', stacklevel=2)
        return self.config.RevealHelpPreprocessor.url_prefix
    return 'https://unpkg.com/reveal.js@4.0.2'