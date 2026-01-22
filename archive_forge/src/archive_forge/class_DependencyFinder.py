import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
class DependencyFinder(object):
    """
    Locate dependencies for distributions.
    """

    def __init__(self, locator=None):
        """
        Initialise an instance, using the specified locator
        to locate distributions.
        """
        self.locator = locator or default_locator
        self.scheme = get_scheme(self.locator.scheme)

    def add_distribution(self, dist):
        """
        Add a distribution to the finder. This will update internal information
        about who provides what.
        :param dist: The distribution to add.
        """
        logger.debug('adding distribution %s', dist)
        name = dist.key
        self.dists_by_name[name] = dist
        self.dists[name, dist.version] = dist
        for p in dist.provides:
            name, version = parse_name_and_version(p)
            logger.debug('Add to provided: %s, %s, %s', name, version, dist)
            self.provided.setdefault(name, set()).add((version, dist))

    def remove_distribution(self, dist):
        """
        Remove a distribution from the finder. This will update internal
        information about who provides what.
        :param dist: The distribution to remove.
        """
        logger.debug('removing distribution %s', dist)
        name = dist.key
        del self.dists_by_name[name]
        del self.dists[name, dist.version]
        for p in dist.provides:
            name, version = parse_name_and_version(p)
            logger.debug('Remove from provided: %s, %s, %s', name, version, dist)
            s = self.provided[name]
            s.remove((version, dist))
            if not s:
                del self.provided[name]

    def get_matcher(self, reqt):
        """
        Get a version matcher for a requirement.
        :param reqt: The requirement
        :type reqt: str
        :return: A version matcher (an instance of
                 :class:`distlib.version.Matcher`).
        """
        try:
            matcher = self.scheme.matcher(reqt)
        except UnsupportedVersionError:
            name = reqt.split()[0]
            matcher = self.scheme.matcher(name)
        return matcher

    def find_providers(self, reqt):
        """
        Find the distributions which can fulfill a requirement.

        :param reqt: The requirement.
         :type reqt: str
        :return: A set of distribution which can fulfill the requirement.
        """
        matcher = self.get_matcher(reqt)
        name = matcher.key
        result = set()
        provided = self.provided
        if name in provided:
            for version, provider in provided[name]:
                try:
                    match = matcher.match(version)
                except UnsupportedVersionError:
                    match = False
                if match:
                    result.add(provider)
                    break
        return result

    def try_to_replace(self, provider, other, problems):
        """
        Attempt to replace one provider with another. This is typically used
        when resolving dependencies from multiple sources, e.g. A requires
        (B >= 1.0) while C requires (B >= 1.1).

        For successful replacement, ``provider`` must meet all the requirements
        which ``other`` fulfills.

        :param provider: The provider we are trying to replace with.
        :param other: The provider we're trying to replace.
        :param problems: If False is returned, this will contain what
                         problems prevented replacement. This is currently
                         a tuple of the literal string 'cantreplace',
                         ``provider``, ``other``  and the set of requirements
                         that ``provider`` couldn't fulfill.
        :return: True if we can replace ``other`` with ``provider``, else
                 False.
        """
        rlist = self.reqts[other]
        unmatched = set()
        for s in rlist:
            matcher = self.get_matcher(s)
            if not matcher.match(provider.version):
                unmatched.add(s)
        if unmatched:
            problems.add(('cantreplace', provider, other, frozenset(unmatched)))
            result = False
        else:
            self.remove_distribution(other)
            del self.reqts[other]
            for s in rlist:
                self.reqts.setdefault(provider, set()).add(s)
            self.add_distribution(provider)
            result = True
        return result

    def find(self, requirement, meta_extras=None, prereleases=False):
        """
        Find a distribution and all distributions it depends on.

        :param requirement: The requirement specifying the distribution to
                            find, or a Distribution instance.
        :param meta_extras: A list of meta extras such as :test:, :build: and
                            so on.
        :param prereleases: If ``True``, allow pre-release versions to be
                            returned - otherwise, don't return prereleases
                            unless they're all that's available.

        Return a set of :class:`Distribution` instances and a set of
        problems.

        The distributions returned should be such that they have the
        :attr:`required` attribute set to ``True`` if they were
        from the ``requirement`` passed to ``find()``, and they have the
        :attr:`build_time_dependency` attribute set to ``True`` unless they
        are post-installation dependencies of the ``requirement``.

        The problems should be a tuple consisting of the string
        ``'unsatisfied'`` and the requirement which couldn't be satisfied
        by any distribution known to the locator.
        """
        self.provided = {}
        self.dists = {}
        self.dists_by_name = {}
        self.reqts = {}
        meta_extras = set(meta_extras or [])
        if ':*:' in meta_extras:
            meta_extras.remove(':*:')
            meta_extras |= set([':test:', ':build:', ':dev:'])
        if isinstance(requirement, Distribution):
            dist = odist = requirement
            logger.debug('passed %s as requirement', odist)
        else:
            dist = odist = self.locator.locate(requirement, prereleases=prereleases)
            if dist is None:
                raise DistlibException('Unable to locate %r' % requirement)
            logger.debug('located %s', odist)
        dist.requested = True
        problems = set()
        todo = set([dist])
        install_dists = set([odist])
        while todo:
            dist = todo.pop()
            name = dist.key
            if name not in self.dists_by_name:
                self.add_distribution(dist)
            else:
                other = self.dists_by_name[name]
                if other != dist:
                    self.try_to_replace(dist, other, problems)
            ireqts = dist.run_requires | dist.meta_requires
            sreqts = dist.build_requires
            ereqts = set()
            if meta_extras and dist in install_dists:
                for key in ('test', 'build', 'dev'):
                    e = ':%s:' % key
                    if e in meta_extras:
                        ereqts |= getattr(dist, '%s_requires' % key)
            all_reqts = ireqts | sreqts | ereqts
            for r in all_reqts:
                providers = self.find_providers(r)
                if not providers:
                    logger.debug('No providers found for %r', r)
                    provider = self.locator.locate(r, prereleases=prereleases)
                    if provider is None and (not prereleases):
                        provider = self.locator.locate(r, prereleases=True)
                    if provider is None:
                        logger.debug('Cannot satisfy %r', r)
                        problems.add(('unsatisfied', r))
                    else:
                        n, v = (provider.key, provider.version)
                        if (n, v) not in self.dists:
                            todo.add(provider)
                        providers.add(provider)
                        if r in ireqts and dist in install_dists:
                            install_dists.add(provider)
                            logger.debug('Adding %s to install_dists', provider.name_and_version)
                for p in providers:
                    name = p.key
                    if name not in self.dists_by_name:
                        self.reqts.setdefault(p, set()).add(r)
                    else:
                        other = self.dists_by_name[name]
                        if other != p:
                            self.try_to_replace(p, other, problems)
        dists = set(self.dists.values())
        for dist in dists:
            dist.build_time_dependency = dist not in install_dists
            if dist.build_time_dependency:
                logger.debug('%s is a build-time dependency only.', dist.name_and_version)
        logger.debug('find done for %s', odist)
        return (dists, problems)