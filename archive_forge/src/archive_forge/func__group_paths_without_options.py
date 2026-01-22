from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def _group_paths_without_options(cls, line_parse_result):
    """
        Given a parsed options specification as a list of groups, combine
        groups without options with the first subsequent group which has
        options.
        A line of the form
            'A B C [opts] D E [opts_2]'
        results in
            [({A, B, C}, [opts]), ({D, E}, [opts_2])]
        """
    active_pathspecs = set()
    for group in line_parse_result:
        active_pathspecs.add(group['pathspec'])
        has_options = 'norm_options' in group or 'plot_options' in group or 'style_options' in group
        if has_options:
            yield (active_pathspecs, group)
            active_pathspecs = set()
    if active_pathspecs:
        yield (active_pathspecs, {})