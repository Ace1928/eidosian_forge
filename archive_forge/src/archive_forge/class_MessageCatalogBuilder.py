from codecs import open
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, tzinfo
from os import getenv, path, walk
from time import time
from typing import (Any, DefaultDict, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from uuid import uuid4
from docutils import nodes
from docutils.nodes import Element
from sphinx import addnodes, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains.python import pairindextypes
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging, split_index_msg, status_iterator
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogInfo, docname_to_domain
from sphinx.util.nodes import extract_messages, traverse_translatable_index
from sphinx.util.osutil import canon_path, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.template import SphinxRenderer
class MessageCatalogBuilder(I18nBuilder):
    """
    Builds gettext-style message catalogs (.pot files).
    """
    name = 'gettext'
    epilog = __('The message catalogs are in %(outdir)s.')

    def init(self) -> None:
        super().init()
        self.create_template_bridge()
        self.templates.init(self)

    def _collect_templates(self) -> Set[str]:
        template_files = set()
        for template_path in self.config.templates_path:
            tmpl_abs_path = path.join(self.app.srcdir, template_path)
            for dirpath, _dirs, files in walk(tmpl_abs_path):
                for fn in files:
                    if fn.endswith('.html'):
                        filename = canon_path(path.join(dirpath, fn))
                        template_files.add(filename)
        return template_files

    def _extract_from_template(self) -> None:
        files = list(self._collect_templates())
        files.sort()
        logger.info(bold(__('building [%s]: ') % self.name), nonl=True)
        logger.info(__('targets for %d template files'), len(files))
        extract_translations = self.templates.environment.extract_translations
        for template in status_iterator(files, __('reading templates... '), 'purple', len(files), self.app.verbosity):
            try:
                with open(template, encoding='utf-8') as f:
                    context = f.read()
                for line, _meth, msg in extract_translations(context):
                    origin = MsgOrigin(template, line)
                    self.catalogs['sphinx'].add(msg, origin)
            except Exception as exc:
                raise ThemeError('%s: %r' % (template, exc)) from exc

    def build(self, docnames: Iterable[str], summary: Optional[str]=None, method: str='update') -> None:
        self._extract_from_template()
        super().build(docnames, summary, method)

    def finish(self) -> None:
        super().finish()
        context = {'version': self.config.version, 'copyright': self.config.copyright, 'project': self.config.project, 'last_translator': self.config.gettext_last_translator, 'language_team': self.config.gettext_language_team, 'ctime': datetime.fromtimestamp(timestamp, ltz).strftime('%Y-%m-%d %H:%M%z'), 'display_location': self.config.gettext_location, 'display_uuid': self.config.gettext_uuid}
        for textdomain, catalog in status_iterator(self.catalogs.items(), __('writing message catalogs... '), 'darkgreen', len(self.catalogs), self.app.verbosity, lambda textdomain__: textdomain__[0]):
            ensuredir(path.join(self.outdir, path.dirname(textdomain)))
            context['messages'] = list(catalog)
            content = GettextRenderer(outdir=self.outdir).render('message.pot_t', context)
            pofn = path.join(self.outdir, textdomain + '.pot')
            if should_write(pofn, content):
                with open(pofn, 'w', encoding='utf-8') as pofile:
                    pofile.write(content)