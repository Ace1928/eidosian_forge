import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
Test behaviour when log likelihood cannot be retrieved.

        If log_likelihood=True there is a warning to say log_likelihood group is skipped,
        if log_likelihood=False there is no warning and log_likelihood is skipped.
        