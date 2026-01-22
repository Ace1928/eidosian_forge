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
def _create_element_from_result(domain: Domain, inv_name: Optional[str], data: InventoryItem, node: pending_xref, contnode: TextElement) -> Element:
    proj, version, uri, dispname = data
    if '://' not in uri and node.get('refdoc'):
        uri = path.join(relative_path(node['refdoc'], '.'), uri)
    if version:
        reftitle = _('(in %s v%s)') % (proj, version)
    else:
        reftitle = _('(in %s)') % (proj,)
    newnode = nodes.reference('', '', internal=False, refuri=uri, reftitle=reftitle)
    if node.get('refexplicit'):
        newnode.append(contnode)
    elif dispname == '-' or (domain.name == 'std' and node['reftype'] == 'keyword'):
        title = contnode.astext()
        if inv_name is not None and title.startswith(inv_name + ':'):
            newnode.append(contnode.__class__(title[len(inv_name) + 1:], title[len(inv_name) + 1:]))
        else:
            newnode.append(contnode)
    else:
        newnode.append(contnode.__class__(dispname, dispname))
    return newnode