from collections.abc import Mapping
from pathlib import Path
import pytest
from jsonschema_specifications import REGISTRY

    Ignore files like .DS_Store if someone has actually caused one to exist.

    We test here through the private interface as of course the global has
    already loaded our schemas.
    