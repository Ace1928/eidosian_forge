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
class ASTParametersQualifiers(ASTBase):

    def __init__(self, args: List[ASTFunctionParameter], volatile: bool, const: bool, refQual: Optional[str], exceptionSpec: ASTNoexceptSpec, trailingReturn: 'ASTType', override: bool, final: bool, attrs: ASTAttributeList, initializer: Optional[str]) -> None:
        self.args = args
        self.volatile = volatile
        self.const = const
        self.refQual = refQual
        self.exceptionSpec = exceptionSpec
        self.trailingReturn = trailingReturn
        self.override = override
        self.final = final
        self.attrs = attrs
        self.initializer = initializer

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.args

    def get_modifiers_id(self, version: int) -> str:
        res = []
        if self.volatile:
            res.append('V')
        if self.const:
            if version == 1:
                res.append('C')
            else:
                res.append('K')
        if self.refQual == '&&':
            res.append('O')
        elif self.refQual == '&':
            res.append('R')
        return ''.join(res)

    def get_param_id(self, version: int) -> str:
        if version == 1:
            if len(self.args) == 0:
                return ''
            else:
                return '__' + '.'.join((a.get_id(version) for a in self.args))
        if len(self.args) == 0:
            return 'v'
        else:
            return ''.join((a.get_id(version) for a in self.args))

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append('(')
        first = True
        for a in self.args:
            if not first:
                res.append(', ')
            first = False
            res.append(str(a))
        res.append(')')
        if self.volatile:
            res.append(' volatile')
        if self.const:
            res.append(' const')
        if self.refQual:
            res.append(' ')
            res.append(self.refQual)
        if self.exceptionSpec:
            res.append(' ')
            res.append(transform(self.exceptionSpec))
        if self.trailingReturn:
            res.append(' -> ')
            res.append(transform(self.trailingReturn))
        if self.final:
            res.append(' final')
        if self.override:
            res.append(' override')
        if len(self.attrs) != 0:
            res.append(' ')
            res.append(transform(self.attrs))
        if self.initializer:
            res.append(' = ')
            res.append(self.initializer)
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if mode == 'lastIsName':
            paramlist = addnodes.desc_parameterlist()
            for arg in self.args:
                param = addnodes.desc_parameter('', '', noemph=True)
                arg.describe_signature(param, 'param', env, symbol=symbol)
                paramlist += param
            signode += paramlist
        else:
            signode += addnodes.desc_sig_punctuation('(', '(')
            first = True
            for arg in self.args:
                if not first:
                    signode += addnodes.desc_sig_punctuation(',', ',')
                    signode += addnodes.desc_sig_space()
                first = False
                arg.describe_signature(signode, 'markType', env, symbol=symbol)
            signode += addnodes.desc_sig_punctuation(')', ')')

        def _add_anno(signode: TextElement, text: str) -> None:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_keyword(text, text)
        if self.volatile:
            _add_anno(signode, 'volatile')
        if self.const:
            _add_anno(signode, 'const')
        if self.refQual:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation(self.refQual, self.refQual)
        if self.exceptionSpec:
            signode += addnodes.desc_sig_space()
            self.exceptionSpec.describe_signature(signode, mode, env, symbol)
        if self.trailingReturn:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_operator('->', '->')
            signode += addnodes.desc_sig_space()
            self.trailingReturn.describe_signature(signode, mode, env, symbol)
        if self.final:
            _add_anno(signode, 'final')
        if self.override:
            _add_anno(signode, 'override')
        if len(self.attrs) != 0:
            signode += addnodes.desc_sig_space()
            self.attrs.describe_signature(signode)
        if self.initializer:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation('=', '=')
            signode += addnodes.desc_sig_space()
            assert self.initializer in ('0', 'delete', 'default')
            if self.initializer == '0':
                signode += addnodes.desc_sig_literal_number('0', '0')
            else:
                signode += addnodes.desc_sig_keyword(self.initializer, self.initializer)