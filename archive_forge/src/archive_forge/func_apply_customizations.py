import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def apply_customizations(cls, spec, options):
    """
        Apply the given option specs to the supplied options tree.
        """
    for key in sorted(spec.keys()):
        if isinstance(spec[key], (list, tuple)):
            customization = {v.key: v for v in spec[key]}
        else:
            customization = {k: Options(**v) if isinstance(v, dict) else v for k, v in spec[key].items()}
        customization = {k: v.keywords_target(key.split('.')[0]) for k, v in customization.items()}
        options[str(key)] = customization
    return options