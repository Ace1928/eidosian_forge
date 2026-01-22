from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
class PathAliases:
    """A collection of aliases for paths.

    When combining data files from remote machines, often the paths to source
    code are different, for example, due to OS differences, or because of
    serialized checkouts on continuous integration machines.

    A `PathAliases` object tracks a list of pattern/result pairs, and can
    map a path through those aliases to produce a unified path.

    """

    def __init__(self, debugfn: Callable[[str], None] | None=None, relative: bool=False) -> None:
        self.aliases: list[tuple[str, re.Pattern[str], str]] = []
        self.debugfn = debugfn or (lambda msg: 0)
        self.relative = relative
        self.pprinted = False

    def pprint(self) -> None:
        """Dump the important parts of the PathAliases, for debugging."""
        self.debugfn(f'Aliases (relative={self.relative}):')
        for original_pattern, regex, result in self.aliases:
            self.debugfn(f' Rule: {original_pattern!r} -> {result!r} using regex {regex.pattern!r}')

    def add(self, pattern: str, result: str) -> None:
        """Add the `pattern`/`result` pair to the list of aliases.

        `pattern` is an `glob`-style pattern.  `result` is a simple
        string.  When mapping paths, if a path starts with a match against
        `pattern`, then that match is replaced with `result`.  This models
        isomorphic source trees being rooted at different places on two
        different machines.

        `pattern` can't end with a wildcard component, since that would
        match an entire tree, and not just its root.

        """
        original_pattern = pattern
        pattern_sep = sep(pattern)
        if len(pattern) > 1:
            pattern = pattern.rstrip('\\/')
        if pattern.endswith('*'):
            raise ConfigError('Pattern must not end with wildcards.')
        if not self.relative:
            if not pattern.startswith('*') and (not isabs_anywhere(pattern + pattern_sep)):
                pattern = abs_file(pattern)
        if not pattern.endswith(pattern_sep):
            pattern += pattern_sep
        regex = globs_to_regex([pattern], case_insensitive=True, partial=True)
        result_sep = sep(result)
        result = result.rstrip('\\/') + result_sep
        self.aliases.append((original_pattern, regex, result))

    def map(self, path: str, exists: Callable[[str], bool]=source_exists) -> str:
        """Map `path` through the aliases.

        `path` is checked against all of the patterns.  The first pattern to
        match is used to replace the root of the path with the result root.
        Only one pattern is ever used.  If no patterns match, `path` is
        returned unchanged.

        The separator style in the result is made to match that of the result
        in the alias.

        `exists` is a function to determine if the resulting path actually
        exists.

        Returns the mapped path.  If a mapping has happened, this is a
        canonical path.  If no mapping has happened, it is the original value
        of `path` unchanged.

        """
        if not self.pprinted:
            self.pprint()
            self.pprinted = True
        for original_pattern, regex, result in self.aliases:
            if (m := regex.match(path)):
                new = path.replace(m[0], result)
                new = new.replace(sep(path), sep(result))
                if not self.relative:
                    new = canonical_filename(new)
                dot_start = result.startswith(('./', '.\\')) and len(result) > 2
                if new.startswith(('./', '.\\')) and (not dot_start):
                    new = new[2:]
                if not exists(new):
                    self.debugfn(f'Rule {original_pattern!r} changed {path!r} to {new!r} ' + "which doesn't exist, continuing")
                    continue
                self.debugfn(f'Matched path {path!r} to rule {original_pattern!r} -> {result!r}, ' + f'producing {new!r}')
                return new
        if self.relative:
            path = relative_filename(path)
        if self.relative and (not isabs_anywhere(path)):
            parts = re.split('[/\\\\]', path)
            if len(parts) > 1:
                dir1 = parts[0]
                pattern = f'*/{dir1}'
                regex_pat = f'^(.*[\\\\/])?{re.escape(dir1)}[\\\\/]'
                result = f'{dir1}{os.sep}'
                if not any((p == pattern for p, _, _ in self.aliases)):
                    self.debugfn(f'Generating rule: {pattern!r} -> {result!r} using regex {regex_pat!r}')
                    self.aliases.append((pattern, re.compile(regex_pat), result))
                    return self.map(path, exists=exists)
        self.debugfn(f'No rules match, path {path!r} is unchanged')
        return path