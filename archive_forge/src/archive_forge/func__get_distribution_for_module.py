import abc
import inspect
from stevedore import extension
from cliff import _argparse
def _get_distribution_for_module(module):
    """Return the distribution containing the module."""
    dist_name = None
    if module:
        pkg_name = module.__name__.partition('.')[0]
        dist_name = _get_distributions_by_modules().get(pkg_name)
    return dist_name