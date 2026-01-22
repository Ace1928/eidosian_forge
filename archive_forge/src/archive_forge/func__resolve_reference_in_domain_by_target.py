import concurrent.futures
import functools
import posixpath
import re
import sys
import time
from os import path
from types import ModuleType
from typing import IO, Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit, urlunsplit
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.utils import Reporter, relative_path
import sphinx
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders.html import INVENTORY_FILENAME
from sphinx.config import Config
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import _, __
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging, requests
from sphinx.util.docutils import CustomReSTDispatcher, SphinxRole
from sphinx.util.inventory import InventoryFile
from sphinx.util.typing import Inventory, InventoryItem, RoleFunction
def _resolve_reference_in_domain_by_target(inv_name: Optional[str], inventory: Inventory, domain: Domain, objtypes: List[str], target: str, node: pending_xref, contnode: TextElement) -> Optional[Element]:
    for objtype in objtypes:
        if objtype not in inventory:
            continue
        if target in inventory[objtype]:
            data = inventory[objtype][target]
        elif objtype == 'std:term':
            target_lower = target.lower()
            insensitive_matches = list(filter(lambda k: k.lower() == target_lower, inventory[objtype].keys()))
            if insensitive_matches:
                data = inventory[objtype][insensitive_matches[0]]
            else:
                continue
        else:
            continue
        return _create_element_from_result(domain, inv_name, data, node, contnode)
    return None