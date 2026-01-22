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
def add_view_copy_derivatives(infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], view_groups: List[NativeFunctionsViewGroup]) -> None:
    view_name_to_group: Dict[OperatorName, NativeFunctionsViewGroup] = {g.view.func.name: g for g in view_groups}
    view_infos = {}
    for info_dispatch_dict in infos.values():
        maybe_view_group = None
        view_copy_differentiability_infos = {}
        for dispatch_key, info in info_dispatch_dict.items():
            maybe_view_group = view_name_to_group.get(info.func.func.name, None)
            if maybe_view_group is not None and maybe_view_group.view_copy is not None:
                view_copy_info = info.create_view_copy_from_view_derivative(maybe_view_group)
                if view_copy_info is not None:
                    fn_schema = view_copy_info.func.func
                    view_copy_differentiability_infos[dispatch_key] = view_copy_info
            else:
                break
        if len(view_copy_differentiability_infos) > 0:
            assert fn_schema is not None
            view_infos[fn_schema] = view_copy_differentiability_infos
    infos.update(view_infos)