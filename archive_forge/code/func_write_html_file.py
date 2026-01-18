from __future__ import annotations
import collections
import datetime
import functools
import json
import os
import re
import shutil
import string
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING, cast
import coverage
from coverage.data import CoverageData, add_data_to_hash
from coverage.exceptions import NoDataError
from coverage.files import flat_rootname
from coverage.misc import ensure_dir, file_be_gone, Hasher, isolate_module, format_local_datetime
from coverage.misc import human_sorted, plural, stdout_link
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.templite import Templite
from coverage.types import TLineNo, TMorf
from coverage.version import __url__
def write_html_file(self, ftr: FileToReport, prev_html: str, next_html: str) -> None:
    """Generate an HTML file for one source file."""
    self.make_directory()
    if self.incr.can_skip_file(self.data, ftr.fr, ftr.rootname):
        self.file_summaries.append(self.incr.index_info(ftr.rootname))
        return
    file_data = self.datagen.data_for_file(ftr.fr, ftr.analysis)
    contexts = collections.Counter((c for cline in file_data.lines for c in cline.contexts))
    context_codes = {y: i for i, y in enumerate((x[0] for x in contexts.most_common()))}
    if context_codes:
        contexts_json = json.dumps({encode_int(v): k for k, v in context_codes.items()}, indent=2)
    else:
        contexts_json = None
    for ldata in file_data.lines:
        html_parts = []
        for tok_type, tok_text in ldata.tokens:
            if tok_type == 'ws':
                html_parts.append(escape(tok_text))
            else:
                tok_html = escape(tok_text) or '&nbsp;'
                html_parts.append(f'<span class="{tok_type}">{tok_html}</span>')
        ldata.html = ''.join(html_parts)
        if ldata.context_list:
            encoded_contexts = [encode_int(context_codes[c_context]) for c_context in ldata.context_list]
            code_width = max((len(ec) for ec in encoded_contexts))
            ldata.context_str = str(code_width) + ''.join((ec.ljust(code_width) for ec in encoded_contexts))
        else:
            ldata.context_str = ''
        if ldata.short_annotations:
            ldata.annotate = ',&nbsp;&nbsp; '.join((f'{ldata.number}&#x202F;&#x219B;&#x202F;{d}' for d in ldata.short_annotations))
        else:
            ldata.annotate = None
        if ldata.long_annotations:
            longs = ldata.long_annotations
            if len(longs) == 1:
                ldata.annotate_long = longs[0]
            else:
                ldata.annotate_long = '{:d} missed branches: {}'.format(len(longs), ', '.join((f'{num:d}) {ann_long}' for num, ann_long in enumerate(longs, start=1))))
        else:
            ldata.annotate_long = None
        css_classes = []
        if ldata.category:
            css_classes.append(self.template_globals['category'][ldata.category])
        ldata.css_class = ' '.join(css_classes) or 'pln'
    html_path = os.path.join(self.directory, ftr.html_filename)
    html = self.source_tmpl.render({**file_data.__dict__, 'contexts_json': contexts_json, 'prev_html': prev_html, 'next_html': next_html})
    write_html(html_path, html)
    index_info: IndexInfoDict = {'nums': ftr.analysis.numbers, 'html_filename': ftr.html_filename, 'relative_filename': ftr.fr.relative_filename()}
    self.file_summaries.append(index_info)
    self.incr.set_index_info(ftr.rootname, index_info)