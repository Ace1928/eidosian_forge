import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class CPPFunctionObject(CPPObject):
    object_type = 'function'
    doc_field_types = CPPObject.doc_field_types + [GroupedField('parameter', label=_('Parameters'), names=('param', 'parameter', 'arg', 'argument'), can_collapse=True), GroupedField('exceptions', label=_('Throws'), rolename='expr', names=('throws', 'throw', 'exception'), can_collapse=True), GroupedField('retval', label=_('Return values'), names=('retvals', 'retval'), can_collapse=True), Field('returnvalue', label=_('Returns'), has_arg=False, names=('returns', 'return'))]