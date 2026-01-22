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
class HtmlDataGeneration:
    """Generate structured data to be turned into HTML reports."""
    EMPTY = '(empty)'

    def __init__(self, cov: Coverage) -> None:
        self.coverage = cov
        self.config = self.coverage.config
        data = self.coverage.get_data()
        self.has_arcs = data.has_arcs()
        if self.config.show_contexts:
            if data.measured_contexts() == {''}:
                self.coverage._warn('No contexts were measured')
        data.set_query_contexts(self.config.report_contexts)

    def data_for_file(self, fr: FileReporter, analysis: Analysis) -> FileData:
        """Produce the data needed for one file's report."""
        if self.has_arcs:
            missing_branch_arcs = analysis.missing_branch_arcs()
            arcs_executed = analysis.arcs_executed()
        if self.config.show_contexts:
            contexts_by_lineno = analysis.data.contexts_by_lineno(analysis.filename)
        lines = []
        for lineno, tokens in enumerate(fr.source_token_lines(), start=1):
            category = ''
            short_annotations = []
            long_annotations = []
            if lineno in analysis.excluded:
                category = 'exc'
            elif lineno in analysis.missing:
                category = 'mis'
            elif self.has_arcs and lineno in missing_branch_arcs:
                category = 'par'
                for b in missing_branch_arcs[lineno]:
                    if b < 0:
                        short_annotations.append('exit')
                    else:
                        short_annotations.append(str(b))
                    long_annotations.append(fr.missing_arc_description(lineno, b, arcs_executed))
            elif lineno in analysis.statements:
                category = 'run'
            contexts = []
            contexts_label = ''
            context_list = []
            if category and self.config.show_contexts:
                contexts = human_sorted((c or self.EMPTY for c in contexts_by_lineno.get(lineno, ())))
                if contexts == [self.EMPTY]:
                    contexts_label = self.EMPTY
                else:
                    contexts_label = f'{len(contexts)} ctx'
                    context_list = contexts
            lines.append(LineData(tokens=tokens, number=lineno, category=category, statement=lineno in analysis.statements, contexts=contexts, contexts_label=contexts_label, context_list=context_list, short_annotations=short_annotations, long_annotations=long_annotations))
        file_data = FileData(relative_filename=fr.relative_filename(), nums=analysis.numbers, lines=lines)
        return file_data