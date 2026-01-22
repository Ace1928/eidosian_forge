import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
@property
def _trainer_resources_not_none(self):
    if self.trainer_resources is None:
        if self.num_workers:
            try:
                import google.colab
                trainer_resources = 0
            except ImportError:
                trainer_resources = 1
        else:
            trainer_resources = 1
        return {'CPU': trainer_resources}
    return {k: v for k, v in self.trainer_resources.items() if v != 0}