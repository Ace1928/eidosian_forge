import yaml
from breezy import errors, hooks
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
Hooks for yaml-formatted version-info output.