import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set
def _traverse_runnable(lc_model, dependency_dict: DefaultDict[str, List[Any]], visited: Set[str]):
    """
    This function contains the logic to traverse a langchain_core.runnables.RunnableSerializable
    object. It first inspects the current object using _extract_dependency_dict_from_lc_model
    and then, if the current object is a Runnable, it recursively inspects its children returned
    by lc_model.get_graph().nodes.values().
    This function supports arbitrary LCEL chain.
    """
    from langchain_core.runnables import Runnable
    current_object_id = id(lc_model)
    if current_object_id in visited:
        return
    visited.add(current_object_id)
    _extract_dependency_dict_from_lc_model(lc_model, dependency_dict)
    if isinstance(lc_model, Runnable):
        for node in lc_model.get_graph().nodes.values():
            _traverse_runnable(node.data, dependency_dict, visited)
    else:
        pass
    return