import yaml
from breezy import errors, hooks
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
class YamlVersionInfoBuilderHooks(hooks.Hooks):
    """Hooks for yaml-formatted version-info output."""

    def __init__(self):
        super().__init__('breezy.version_info_formats.format_yaml', 'YamlVersionInfoBuilder.hooks')
        self.add_hook('revision', 'Invoked when adding information about a revision to the YAML stanza that is printed. revision is called with a revision object and a YAML stanza.', (3, 3))