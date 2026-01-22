from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
class DistributionPath(object):
    """
    Represents a set of distributions installed on a path (typically sys.path).
    """

    def __init__(self, path=None, include_egg=False):
        """
        Create an instance from a path, optionally including legacy (distutils/
        setuptools/distribute) distributions.
        :param path: The path to use, as a list of directories. If not specified,
                     sys.path is used.
        :param include_egg: If True, this instance will look for and return legacy
                            distributions as well as those based on PEP 376.
        """
        if path is None:
            path = sys.path
        self.path = path
        self._include_dist = True
        self._include_egg = include_egg
        self._cache = _Cache()
        self._cache_egg = _Cache()
        self._cache_enabled = True
        self._scheme = get_scheme('default')

    def _get_cache_enabled(self):
        return self._cache_enabled

    def _set_cache_enabled(self, value):
        self._cache_enabled = value
    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)

    def clear_cache(self):
        """
        Clears the internal cache.
        """
        self._cache.clear()
        self._cache_egg.clear()

    def _yield_distributions(self):
        """
        Yield .dist-info and/or .egg(-info) distributions.
        """
        seen = set()
        for path in self.path:
            finder = resources.finder_for_path(path)
            if finder is None:
                continue
            r = finder.find('')
            if not r or not r.is_container:
                continue
            rset = sorted(r.resources)
            for entry in rset:
                r = finder.find(entry)
                if not r or r.path in seen:
                    continue
                try:
                    if self._include_dist and entry.endswith(DISTINFO_EXT):
                        possible_filenames = [METADATA_FILENAME, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME]
                        for metadata_filename in possible_filenames:
                            metadata_path = posixpath.join(entry, metadata_filename)
                            pydist = finder.find(metadata_path)
                            if pydist:
                                break
                        else:
                            continue
                        with contextlib.closing(pydist.as_stream()) as stream:
                            metadata = Metadata(fileobj=stream, scheme='legacy')
                        logger.debug('Found %s', r.path)
                        seen.add(r.path)
                        yield new_dist_class(r.path, metadata=metadata, env=self)
                    elif self._include_egg and entry.endswith(('.egg-info', '.egg')):
                        logger.debug('Found %s', r.path)
                        seen.add(r.path)
                        yield old_dist_class(r.path, self)
                except Exception as e:
                    msg = 'Unable to read distribution at %s, perhaps due to bad metadata: %s'
                    logger.warning(msg, r.path, e)
                    import warnings
                    warnings.warn(msg % (r.path, e), stacklevel=2)

    def _generate_cache(self):
        """
        Scan the path for distributions and populate the cache with
        those that are found.
        """
        gen_dist = not self._cache.generated
        gen_egg = self._include_egg and (not self._cache_egg.generated)
        if gen_dist or gen_egg:
            for dist in self._yield_distributions():
                if isinstance(dist, InstalledDistribution):
                    self._cache.add(dist)
                else:
                    self._cache_egg.add(dist)
            if gen_dist:
                self._cache.generated = True
            if gen_egg:
                self._cache_egg.generated = True

    @classmethod
    def distinfo_dirname(cls, name, version):
        """
        The *name* and *version* parameters are converted into their
        filename-escaped form, i.e. any ``'-'`` characters are replaced
        with ``'_'`` other than the one in ``'dist-info'`` and the one
        separating the name from the version number.

        :parameter name: is converted to a standard distribution name by replacing
                         any runs of non- alphanumeric characters with a single
                         ``'-'``.
        :type name: string
        :parameter version: is converted to a standard version string. Spaces
                            become dots, and all other non-alphanumeric characters
                            (except dots) become dashes, with runs of multiple
                            dashes condensed to a single dash.
        :type version: string
        :returns: directory name
        :rtype: string"""
        name = name.replace('-', '_')
        return '-'.join([name, version]) + DISTINFO_EXT

    def get_distributions(self):
        """
        Provides an iterator that looks for distributions and returns
        :class:`InstalledDistribution` or
        :class:`EggInfoDistribution` instances for each one of them.

        :rtype: iterator of :class:`InstalledDistribution` and
                :class:`EggInfoDistribution` instances
        """
        if not self._cache_enabled:
            for dist in self._yield_distributions():
                yield dist
        else:
            self._generate_cache()
            for dist in self._cache.path.values():
                yield dist
            if self._include_egg:
                for dist in self._cache_egg.path.values():
                    yield dist

    def get_distribution(self, name):
        """
        Looks for a named distribution on the path.

        This function only returns the first result found, as no more than one
        value is expected. If nothing is found, ``None`` is returned.

        :rtype: :class:`InstalledDistribution`, :class:`EggInfoDistribution`
                or ``None``
        """
        result = None
        name = name.lower()
        if not self._cache_enabled:
            for dist in self._yield_distributions():
                if dist.key == name:
                    result = dist
                    break
        else:
            self._generate_cache()
            if name in self._cache.name:
                result = self._cache.name[name][0]
            elif self._include_egg and name in self._cache_egg.name:
                result = self._cache_egg.name[name][0]
        return result

    def provides_distribution(self, name, version=None):
        """
        Iterates over all distributions to find which distributions provide *name*.
        If a *version* is provided, it will be used to filter the results.

        This function only returns the first result found, since no more than
        one values are expected. If the directory is not found, returns ``None``.

        :parameter version: a version specifier that indicates the version
                            required, conforming to the format in ``PEP-345``

        :type name: string
        :type version: string
        """
        matcher = None
        if version is not None:
            try:
                matcher = self._scheme.matcher('%s (%s)' % (name, version))
            except ValueError:
                raise DistlibException('invalid name or version: %r, %r' % (name, version))
        for dist in self.get_distributions():
            if not hasattr(dist, 'provides'):
                logger.debug('No "provides": %s', dist)
            else:
                provided = dist.provides
                for p in provided:
                    p_name, p_ver = parse_name_and_version(p)
                    if matcher is None:
                        if p_name == name:
                            yield dist
                            break
                    elif p_name == name and matcher.match(p_ver):
                        yield dist
                        break

    def get_file_path(self, name, relative_path):
        """
        Return the path to a resource file.
        """
        dist = self.get_distribution(name)
        if dist is None:
            raise LookupError('no distribution named %r found' % name)
        return dist.get_resource_path(relative_path)

    def get_exported_entries(self, category, name=None):
        """
        Return all of the exported entries in a particular category.

        :param category: The category to search for entries.
        :param name: If specified, only entries with that name are returned.
        """
        for dist in self.get_distributions():
            r = dist.exports
            if category in r:
                d = r[category]
                if name is not None:
                    if name in d:
                        yield d[name]
                else:
                    for v in d.values():
                        yield v