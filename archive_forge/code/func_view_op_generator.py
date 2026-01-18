import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def view_op_generator(self, g: NativeFunctionsViewGroup, backend_index: BackendIndex) -> str:
    schema = str(g.view.func)
    populated_argument = generate_arg_extraction(g.view.func)
    functional_variant_call = generate_call_to_view_ops(g, backend_index)
    generated = f'\n      if (n->matches(torch::schema("aten::{schema}"))) {{\n        return [](ProcessedNode* p_node) {{\n          {populated_argument}\n            p_node->Output(0) = {functional_variant_call};\n        }};\n      }}'
    return generated