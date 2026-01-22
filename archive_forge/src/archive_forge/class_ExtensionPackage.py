import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union
import tornado
from jupyterlab_server.translation_utils import translator
from traitlets import Enum
from traitlets.config import Configurable, LoggingConfigurable
from jupyterlab.commands import (
@dataclass(frozen=True)
class ExtensionPackage:
    """Extension package entry.

    Attributes:
        name: Package name
        description: Package description
        homepage_url: Package home page
        pkg_type: Type of package - ["prebuilt", "source"]
        allowed: [optional] Whether this extension is allowed or not - default True
        approved: [optional] Whether the package is approved by your administrators - default False
        companion: [optional] Type of companion for the frontend extension - [None, "kernel", "server"]; default None
        core: [optional] Whether the package is a core package or not - default False
        enabled: [optional] Whether the package is enabled or not - default False
        install: [optional] Extension package installation instructions - default None
        installed: [optional] Whether the extension is currently installed - default None
        installed_version: [optional] Installed version - default ""
        latest_version: [optional] Latest available version - default ""
        status: [optional] Package status - ["ok", "warning", "error"]; default "ok"
        author: [optional] Package author - default None
        license: [optional] Package license - default None
        bug_tracker_url: [optional] Package bug tracker URL - default None
        documentation_url: [optional] Package documentation URL - default None
        package_manager_url: Package home page in the package manager - default None
        repository_url: [optional] Package code repository URL - default None
    """
    name: str
    description: str
    homepage_url: str
    pkg_type: str
    allowed: bool = True
    approved: bool = False
    companion: Optional[str] = None
    core: bool = False
    enabled: bool = False
    install: Optional[dict] = None
    installed: Optional[bool] = None
    installed_version: str = ''
    latest_version: str = ''
    status: str = 'ok'
    author: Optional[str] = None
    license: Optional[str] = None
    bug_tracker_url: Optional[str] = None
    documentation_url: Optional[str] = None
    package_manager_url: Optional[str] = None
    repository_url: Optional[str] = None