from ... import \
from ... import config as _mod_config
from ... import version_info  # noqa: F401
from ... import lazy_regex, trace
from ...commands import plugin_cmds
from ...directory_service import directories
from ...help_topics import topic_registry
from ...forge import forges
def _register_directory():
    directories.register_lazy('lp:', 'breezy.plugins.launchpad.lp_directory', 'LaunchpadDirectory', 'Launchpad-based directory service')
    directories.register_lazy('lp+bzr:', 'breezy.plugins.launchpad.lp_directory', 'LaunchpadDirectory', 'Bazaar-specific Launchpad directory service')