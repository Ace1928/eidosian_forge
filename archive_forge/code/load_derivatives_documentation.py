import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader

        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        