from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
def _handle_metadata(trees, notify):
    """ Handle an element for filtering existing metadata.

    Parameters
    ----------
    trees : list of lark.tree.Tree
        The children tree for the "metadata" rule.
        It contains only one item.
    notify : bool
        True if the final target should notify, else False.

    Returns
    -------
    expression : ObserverExpression
    """
    token, = trees
    metadata_name = token.value
    return expression_module.metadata(metadata_name, notify=notify)