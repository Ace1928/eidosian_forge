import os.path
from six.moves import configparser
from sphinx.util import logging
import pbr.version
def _builder_inited(app):
    project_name = _get_project_name(app.srcdir)
    try:
        version_info = pbr.version.VersionInfo(project_name)
    except Exception:
        version_info = None
    if version_info and (not app.config.version) and (not app.config.release):
        app.config.version = version_info.canonical_version_string()
        app.config.release = version_info.version_string_with_vcs()