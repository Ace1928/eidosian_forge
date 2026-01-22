import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def _format_module_infos(formatter, package_name, mod_sequence, target=None):
    """Format modules.
    """
    formatter.title('Listings for {}'.format(package_name))
    alias_map = {}
    for mod in mod_sequence:
        stat = _Stat()
        modname = mod.__name__
        formatter.begin_module_section(formatter.escape(modname))
        for info in inspect_module(mod, target=target, alias=alias_map):
            nbtype = info['numba_type']
            if nbtype is not None:
                stat.supported += 1
                formatter.write_supported_item(modname=formatter.escape(info['module'].__name__), itemname=formatter.escape(info['name']), typename=formatter.escape(str(nbtype)), explained=formatter.escape(info['explained']), sources=info['source_infos'], alias=info.get('alias'))
            else:
                stat.unsupported += 1
                formatter.write_unsupported_item(modname=formatter.escape(info['module'].__name__), itemname=formatter.escape(info['name']))
        formatter.write_statistic(stat)
        formatter.end_module_section()