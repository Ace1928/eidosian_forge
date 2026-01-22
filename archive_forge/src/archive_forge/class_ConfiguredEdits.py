from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
class ConfiguredEdits(AbstractEdits):

    def __init__(self, simple_edits, cut_buffer_edits, awaiting_config, config, key_dispatch):
        super().__init__(dict(simple_edits), dict(cut_buffer_edits))
        for attr, func in awaiting_config.items():
            for key in key_dispatch[getattr(config, attr)]:
                super().add(key, func, overwrite=True)

    def add_config_attr(self, config_attr, func):
        raise NotImplementedError('Config already set on this mapping')

    def add(self, key, func, overwrite=False):
        raise NotImplementedError('Config already set on this mapping')