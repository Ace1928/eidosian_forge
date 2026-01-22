import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
class trace:
    enabled = os.environ.get('TORCH_COMPILE_DEBUG', '0') == '1'
    debug_dir = None
    debug_log = False
    info_log = False
    fx_graph = True
    fx_graph_transformed = True
    ir_pre_fusion = True
    ir_post_fusion = True
    output_code = True
    graph_diagram = os.environ.get('INDUCTOR_POST_FUSION_SVG', '0') == '1'
    draw_orig_fx_graph = os.environ.get('INDUCTOR_ORIG_FX_SVG', '0') == '1'
    dot_graph_shape = os.environ.get('INDUCTOR_DOT_GRAPH_SHAPE_SVG', None)
    compile_profile = False
    upload_tar = None