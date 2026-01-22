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
class IntersphinxRole(SphinxRole):
    _re_inv_ref = re.compile('(\\+([^:]+))?:(.*)')

    def __init__(self, orig_name: str) -> None:
        self.orig_name = orig_name

    def run(self) -> Tuple[List[Node], List[system_message]]:
        assert self.name == self.orig_name.lower()
        inventory, name_suffix = self.get_inventory_and_name_suffix(self.orig_name)
        if inventory and (not inventory_exists(self.env, inventory)):
            logger.warning(__('inventory for external cross-reference not found: %s'), inventory, location=(self.env.docname, self.lineno))
            return ([], [])
        role_name = self.get_role_name(name_suffix)
        if role_name is None:
            logger.warning(__('role for external cross-reference not found: %s'), name_suffix, location=(self.env.docname, self.lineno))
            return ([], [])
        result, messages = self.invoke_role(role_name)
        for node in result:
            if isinstance(node, pending_xref):
                node['intersphinx'] = True
                node['inventory'] = inventory
        return (result, messages)

    def get_inventory_and_name_suffix(self, name: str) -> Tuple[Optional[str], str]:
        assert name.startswith('external'), name
        assert name[8] in ':+', name
        inv, suffix = IntersphinxRole._re_inv_ref.fullmatch(name, 8).group(2, 3)
        return (inv, suffix)

    def get_role_name(self, name: str) -> Optional[Tuple[str, str]]:
        names = name.split(':')
        if len(names) == 1:
            default_domain = self.env.temp_data.get('default_domain')
            domain = default_domain.name if default_domain else None
            role = names[0]
        elif len(names) == 2:
            domain = names[0]
            role = names[1]
        else:
            return None
        if domain and self.is_existent_role(domain, role):
            return (domain, role)
        elif self.is_existent_role('std', role):
            return ('std', role)
        else:
            return None

    def is_existent_role(self, domain_name: str, role_name: str) -> bool:
        try:
            domain = self.env.get_domain(domain_name)
            if role_name in domain.roles:
                return True
            else:
                return False
        except ExtensionError:
            return False

    def invoke_role(self, role: Tuple[str, str]) -> Tuple[List[Node], List[system_message]]:
        domain = self.env.get_domain(role[0])
        if domain:
            role_func = domain.role(role[1])
            return role_func(':'.join(role), self.rawtext, self.text, self.lineno, self.inliner, self.options, self.content)
        else:
            return ([], [])