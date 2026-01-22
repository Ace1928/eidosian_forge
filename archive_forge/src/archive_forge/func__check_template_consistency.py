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
def _check_template_consistency(self, nestedName: ASTNestedName, templatePrefix: ASTTemplateDeclarationPrefix, fullSpecShorthand: bool, isMember: bool=False) -> ASTTemplateDeclarationPrefix:
    numArgs = nestedName.num_templates()
    isMemberInstantiation = False
    if not templatePrefix:
        numParams = 0
    elif isMember and templatePrefix.templates is None:
        numParams = 0
        isMemberInstantiation = True
    else:
        numParams = len(templatePrefix.templates)
    if numArgs + 1 < numParams:
        self.fail('Too few template argument lists comapred to parameter lists. Argument lists: %d, Parameter lists: %d.' % (numArgs, numParams))
    if numArgs > numParams:
        numExtra = numArgs - numParams
        if not fullSpecShorthand and (not isMemberInstantiation):
            msg = 'Too many template argument lists compared to parameter lists. Argument lists: %d, Parameter lists: %d, Extra empty parameters lists prepended: %d.' % (numArgs, numParams, numExtra)
            msg += ' Declaration:\n\t'
            if templatePrefix:
                msg += '%s\n\t' % templatePrefix
            msg += str(nestedName)
            self.warn(msg)
        newTemplates: List[Union[ASTTemplateParams, ASTTemplateIntroduction]] = []
        for _i in range(numExtra):
            newTemplates.append(ASTTemplateParams([], requiresClause=None))
        if templatePrefix and (not isMemberInstantiation):
            newTemplates.extend(templatePrefix.templates)
        templatePrefix = ASTTemplateDeclarationPrefix(newTemplates)
    return templatePrefix