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
class HtmlReporter:
    """HTML reporting."""
    STATIC_FILES = ['style.css', 'coverage_html.js', 'keybd_closed.png', 'keybd_open.png', 'favicon_32.png']

    def __init__(self, cov: Coverage) -> None:
        self.coverage = cov
        self.config = self.coverage.config
        self.directory = self.config.html_dir
        self.skip_covered = self.config.html_skip_covered
        if self.skip_covered is None:
            self.skip_covered = self.config.skip_covered
        self.skip_empty = self.config.html_skip_empty
        if self.skip_empty is None:
            self.skip_empty = self.config.skip_empty
        self.skipped_covered_count = 0
        self.skipped_empty_count = 0
        title = self.config.html_title
        self.extra_css: str | None
        if self.config.extra_css:
            self.extra_css = os.path.basename(self.config.extra_css)
        else:
            self.extra_css = None
        self.data = self.coverage.get_data()
        self.has_arcs = self.data.has_arcs()
        self.file_summaries: list[IndexInfoDict] = []
        self.all_files_nums: list[Numbers] = []
        self.incr = IncrementalChecker(self.directory)
        self.datagen = HtmlDataGeneration(self.coverage)
        self.totals = Numbers(precision=self.config.precision)
        self.directory_was_empty = False
        self.first_fr = None
        self.final_fr = None
        self.template_globals = {'escape': escape, 'pair': pair, 'len': len, '__url__': __url__, '__version__': coverage.__version__, 'title': title, 'time_stamp': format_local_datetime(datetime.datetime.now()), 'extra_css': self.extra_css, 'has_arcs': self.has_arcs, 'show_contexts': self.config.show_contexts, 'category': {'exc': 'exc show_exc', 'mis': 'mis show_mis', 'par': 'par run show_par', 'run': 'run'}}
        self.pyfile_html_source = read_data('pyfile.html')
        self.source_tmpl = Templite(self.pyfile_html_source, self.template_globals)

    def report(self, morfs: Iterable[TMorf] | None) -> float:
        """Generate an HTML report for `morfs`.

        `morfs` is a list of modules or file names.

        """
        self.incr.read()
        self.incr.check_global_data(self.config, self.pyfile_html_source)
        files_to_report = []
        for fr, analysis in get_analysis_to_report(self.coverage, morfs):
            ftr = FileToReport(fr, analysis)
            should = self.should_report_file(ftr)
            if should:
                files_to_report.append(ftr)
            else:
                file_be_gone(os.path.join(self.directory, ftr.html_filename))
        for i, ftr in enumerate(files_to_report):
            if i == 0:
                prev_html = 'index.html'
            else:
                prev_html = files_to_report[i - 1].html_filename
            if i == len(files_to_report) - 1:
                next_html = 'index.html'
            else:
                next_html = files_to_report[i + 1].html_filename
            self.write_html_file(ftr, prev_html, next_html)
        if not self.all_files_nums:
            raise NoDataError('No data to report.')
        self.totals = cast(Numbers, sum(self.all_files_nums))
        if files_to_report:
            first_html = files_to_report[0].html_filename
            final_html = files_to_report[-1].html_filename
        else:
            first_html = final_html = 'index.html'
        self.index_file(first_html, final_html)
        self.make_local_static_report_files()
        return self.totals.n_statements and self.totals.pc_covered

    def make_directory(self) -> None:
        """Make sure our htmlcov directory exists."""
        ensure_dir(self.directory)
        if not os.listdir(self.directory):
            self.directory_was_empty = True

    def make_local_static_report_files(self) -> None:
        """Make local instances of static files for HTML report."""
        for static in self.STATIC_FILES:
            shutil.copyfile(data_filename(static), os.path.join(self.directory, static))
        if self.directory_was_empty:
            with open(os.path.join(self.directory, '.gitignore'), 'w') as fgi:
                fgi.write('# Created by coverage.py\n*\n')
        if self.extra_css:
            assert self.config.extra_css is not None
            shutil.copyfile(self.config.extra_css, os.path.join(self.directory, self.extra_css))

    def should_report_file(self, ftr: FileToReport) -> bool:
        """Determine if we'll report this file."""
        nums = ftr.analysis.numbers
        self.all_files_nums.append(nums)
        if self.skip_covered:
            no_missing_lines = nums.n_missing == 0
            no_missing_branches = nums.n_partial_branches == 0
            if no_missing_lines and no_missing_branches:
                self.skipped_covered_count += 1
                return False
        if self.skip_empty:
            if nums.n_statements == 0:
                self.skipped_empty_count += 1
                return False
        return True

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

    def index_file(self, first_html: str, final_html: str) -> None:
        """Write the index.html file for this report."""
        self.make_directory()
        index_tmpl = Templite(read_data('index.html'), self.template_globals)
        skipped_covered_msg = skipped_empty_msg = ''
        if self.skipped_covered_count:
            n = self.skipped_covered_count
            skipped_covered_msg = f'{n} file{plural(n)} skipped due to complete coverage.'
        if self.skipped_empty_count:
            n = self.skipped_empty_count
            skipped_empty_msg = f'{n} empty file{plural(n)} skipped.'
        html = index_tmpl.render({'files': self.file_summaries, 'totals': self.totals, 'skipped_covered_msg': skipped_covered_msg, 'skipped_empty_msg': skipped_empty_msg, 'first_html': first_html, 'final_html': final_html})
        index_file = os.path.join(self.directory, 'index.html')
        write_html(index_file, html)
        print_href = stdout_link(index_file, f'file://{os.path.abspath(index_file)}')
        self.coverage._message(f'Wrote HTML report to {print_href}')
        self.incr.write()