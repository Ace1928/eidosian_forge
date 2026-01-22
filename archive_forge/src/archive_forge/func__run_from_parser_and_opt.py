import io
import argparse
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, CustomHelpFormatter
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
import parlai.utils.logging as logging
from parlai.core.loader import register_script, SCRIPT_REGISTRY  # noqa: F401
@classmethod
def _run_from_parser_and_opt(cls, opt: Opt, parser: ParlaiParser):
    script = cls(opt)
    script.parser = parser
    return script.run()