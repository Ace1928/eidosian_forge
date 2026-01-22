import re
import sys
from typing import Any, Dict, List, Tuple
from docutils import nodes, utils
from docutils.nodes import Node, system_message
from docutils.parsers.rst.states import Inliner
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging, rst
from sphinx.util.nodes import split_explicit_title
from sphinx.util.typing import RoleFunction
class ExternalLinksChecker(SphinxPostTransform):
    """
    For each external link, check if it can be replaced by an extlink.

    We treat each ``reference`` node without ``internal`` attribute as an external link.
    """
    default_priority = 500

    def run(self, **kwargs: Any) -> None:
        if not self.config.extlinks_detect_hardcoded_links:
            return
        for refnode in self.document.findall(nodes.reference):
            self.check_uri(refnode)

    def check_uri(self, refnode: nodes.reference) -> None:
        """
        If the URI in ``refnode`` has a replacement in ``extlinks``,
        emit a warning with a replacement suggestion.
        """
        if 'internal' in refnode or 'refuri' not in refnode:
            return
        uri = refnode['refuri']
        title = refnode.astext()
        for alias, (base_uri, _caption) in self.app.config.extlinks.items():
            if sys.version_info < (3, 7):
                uri_pattern = re.compile(re.escape(base_uri).replace('\\%s', '(?P<value>.+)'))
            else:
                uri_pattern = re.compile(re.escape(base_uri).replace('%s', '(?P<value>.+)'))
            match = uri_pattern.match(uri)
            if match and match.groupdict().get('value') and ('/' not in match.groupdict()['value']):
                msg = __('hardcoded link %r could be replaced by an extlink (try using %r instead)')
                value = match.groupdict().get('value')
                if uri != title:
                    replacement = f':{alias}:`{rst.escape(title)} <{value}>`'
                else:
                    replacement = f':{alias}:`{value}`'
                logger.warning(msg, uri, replacement, location=refnode)