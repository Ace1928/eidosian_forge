import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _get_node_provider_cls(provider_config: Dict[str, Any]):
    """Get the node provider class for a given provider config.

    Note that this may be used by private node providers that proxy methods to
    built-in node providers, so we should maintain backwards compatibility.

    Args:
        provider_config: provider section of the autoscaler config.

    Returns:
        NodeProvider class
    """
    importer = _NODE_PROVIDERS.get(provider_config['type'])
    if importer is None:
        raise NotImplementedError('Unsupported node provider: {}'.format(provider_config['type']))
    return importer(provider_config)