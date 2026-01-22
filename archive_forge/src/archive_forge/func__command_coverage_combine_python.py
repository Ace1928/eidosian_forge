from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def _command_coverage_combine_python(args: CoverageCombineConfig, host_state: HostState) -> list[str]:
    """Combine Python coverage files and return a list of the output files."""
    coverage = initialize_coverage(args, host_state)
    modules = get_python_modules()
    coverage_files = get_python_coverage_files()

    def _default_stub_value(source_paths: list[str]) -> dict[str, set[tuple[int, int]]]:
        return {path: {(0, 0)} for path in source_paths}
    counter = 0
    sources = _get_coverage_targets(args, walk_compile_targets)
    groups = _build_stub_groups(args, sources, _default_stub_value)
    collection_search_re, collection_sub_re = get_collection_path_regexes()
    for coverage_file in coverage_files:
        counter += 1
        display.info('[%4d/%4d] %s' % (counter, len(coverage_files), coverage_file), verbosity=2)
        group = get_coverage_group(args, coverage_file)
        if group is None:
            display.warning('Unexpected name for coverage file: %s' % coverage_file)
            continue
        for filename, arcs in enumerate_python_arcs(coverage_file, coverage, modules, collection_search_re, collection_sub_re):
            if args.export:
                filename = os.path.relpath(filename)
            if group not in groups:
                groups[group] = {}
            arc_data = groups[group]
            if filename not in arc_data:
                arc_data[filename] = set()
            arc_data[filename].update(arcs)
    output_files = []
    if args.export:
        coverage_file = os.path.join(args.export, '')
        suffix = '=coverage.combined'
    else:
        coverage_file = os.path.join(ResultType.COVERAGE.path, COVERAGE_OUTPUT_FILE_NAME)
        suffix = ''
    path_checker = PathChecker(args, collection_search_re)
    for group in sorted(groups):
        arc_data = groups[group]
        output_file = coverage_file + group + suffix
        if args.explain:
            continue
        updated = coverage.CoverageData(output_file)
        for filename in arc_data:
            if not path_checker.check_path(filename):
                continue
            updated.add_arcs({filename: list(arc_data[filename])})
        if args.all:
            updated.add_arcs(dict(((source[0], []) for source in sources)))
        updated.write()
        if updated:
            output_files.append(output_file)
    path_checker.report()
    return sorted(output_files)