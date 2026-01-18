import os
import platform
import re
from pathlib import Path
import pytest
import sklearn
from sklearn._min_dependencies import dependent_packages
from sklearn.utils.fixes import parse_version
Check versions in pyproject.toml is consistent with _min_dependencies.