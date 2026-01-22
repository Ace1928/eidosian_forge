from __future__ import annotations
import json
import re
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Iterable
import huggingface_hub
import semantic_version as semver
from gradio_client.documentation import document
from huggingface_hub import CommitOperationAdd
from gradio.themes.utils import (
from gradio.themes.utils.readme_content import README_CONTENT
@staticmethod
def _theme_version_exists(space_info: huggingface_hub.hf_api.SpaceInfo, version: str) -> bool:
    assets = get_theme_assets(space_info)
    return any((a.version == semver.Version(version) for a in assets))