from __future__ import (absolute_import, division, print_function)
import functools
import typing as t
from ansible.galaxy.collection.gpg import get_signature_from_source
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import (
from ansible.module_utils.six import string_types
from ansible.utils.version import SemanticVersion, LooseVersion
class CollectionDependencyProviderBase(AbstractProvider):
    """Delegate providing a requirement interface for the resolver."""

    def __init__(self, apis, concrete_artifacts_manager=None, preferred_candidates=None, with_deps=True, with_pre_releases=False, upgrade=False, include_signatures=True):
        """Initialize helper attributes.

        :param api: An instance of the multiple Galaxy APIs wrapper.

        :param concrete_artifacts_manager: An instance of the caching \\
                                           concrete artifacts manager.

        :param with_deps: A flag specifying whether the resolver \\
                          should attempt to pull-in the deps of the \\
                          requested requirements. On by default.

        :param with_pre_releases: A flag specifying whether the \\
                                  resolver should skip pre-releases. \\
                                  Off by default.

        :param upgrade: A flag specifying whether the resolver should \\
                        skip matching versions that are not upgrades. \\
                        Off by default.

        :param include_signatures: A flag to determine whether to retrieve \\
                                   signatures from the Galaxy APIs and \\
                                   include signatures in matching Candidates. \\
                                   On by default.
        """
        self._api_proxy = apis
        self._make_req_from_dict = functools.partial(Requirement.from_requirement_dict, art_mgr=concrete_artifacts_manager)
        self._preferred_candidates = set(preferred_candidates or ())
        self._with_deps = with_deps
        self._with_pre_releases = with_pre_releases
        self._upgrade = upgrade
        self._include_signatures = include_signatures

    def identify(self, requirement_or_candidate):
        """Given requirement or candidate, return an identifier for it.

        This is used to identify a requirement or candidate, e.g.
        whether two requirements should have their specifier parts
        (version ranges or pins) merged, whether two candidates would
        conflict with each other (because they have same name but
        different versions).
        """
        return requirement_or_candidate.canonical_package_id

    def get_preference(self, *args, **kwargs):
        """Return sort key function return value for given requirement.

        This result should be based on preference that is defined as
        "I think this requirement should be resolved first".
        The lower the return value is, the more preferred this
        group of arguments is.

        resolvelib >=0.5.3, <0.7.0

        :param resolution: Currently pinned candidate, or ``None``.

        :param candidates: A list of possible candidates.

        :param information: A list of requirement information.

        Each ``information`` instance is a named tuple with two entries:

          * ``requirement`` specifies a requirement contributing to
            the current candidate list

          * ``parent`` specifies the candidate that provides
            (dependend on) the requirement, or `None`
            to indicate a root requirement.

        resolvelib >=0.7.0, < 0.8.0

        :param identifier: The value returned by ``identify()``.

        :param resolutions: Mapping of identifier, candidate pairs.

        :param candidates: Possible candidates for the identifer.
            Mapping of identifier, list of candidate pairs.

        :param information: Requirement information of each package.
            Mapping of identifier, list of named tuple pairs.
            The named tuples have the entries ``requirement`` and ``parent``.

        resolvelib >=0.8.0, <= 1.0.1

        :param identifier: The value returned by ``identify()``.

        :param resolutions: Mapping of identifier, candidate pairs.

        :param candidates: Possible candidates for the identifer.
            Mapping of identifier, list of candidate pairs.

        :param information: Requirement information of each package.
            Mapping of identifier, list of named tuple pairs.
            The named tuples have the entries ``requirement`` and ``parent``.

        :param backtrack_causes: Sequence of requirement information that were
            the requirements that caused the resolver to most recently backtrack.

        The preference could depend on a various of issues, including
        (not necessarily in this order):

          * Is this package pinned in the current resolution result?

          * How relaxed is the requirement? Stricter ones should
            probably be worked on first? (I don't know, actually.)

          * How many possibilities are there to satisfy this
            requirement? Those with few left should likely be worked on
            first, I guess?

          * Are there any known conflicts for this requirement?
            We should probably work on those with the most
            known conflicts.

        A sortable value should be returned (this will be used as the
        `key` parameter of the built-in sorting function). The smaller
        the value is, the more preferred this requirement is (i.e. the
        sorting function is called with ``reverse=False``).
        """
        raise NotImplementedError

    def _get_preference(self, candidates):
        if any((candidate in self._preferred_candidates for candidate in candidates)):
            return float('-inf')
        return len(candidates)

    def find_matches(self, *args, **kwargs):
        """Find all possible candidates satisfying given requirements.

        This tries to get candidates based on the requirements' types.

        For concrete requirements (SCM, dir, namespace dir, local or
        remote archives), the one-and-only match is returned

        For a "named" requirement, Galaxy-compatible APIs are consulted
        to find concrete candidates for this requirement. Of theres a
        pre-installed candidate, it's prepended in front of others.

        resolvelib >=0.5.3, <0.6.0

        :param requirements: A collection of requirements which all of \\
                             the returned candidates must match. \\
                             All requirements are guaranteed to have \\
                             the same identifier. \\
                             The collection is never empty.

        resolvelib >=0.6.0

        :param identifier: The value returned by ``identify()``.

        :param requirements: The requirements all returned candidates must satisfy.
            Mapping of identifier, iterator of requirement pairs.

        :param incompatibilities: Incompatible versions that must be excluded
            from the returned list.

        :returns: An iterable that orders candidates by preference, \\
                  e.g. the most preferred candidate comes first.
        """
        raise NotImplementedError

    def _find_matches(self, requirements):
        first_req = requirements[0]
        fqcn = first_req.fqcn
        version_req = "A SemVer-compliant version or '*' is required. See https://semver.org to learn how to compose it correctly. "
        version_req += 'This is an issue with the collection.'
        preinstalled_candidates = set()
        if not self._upgrade and first_req.type == 'galaxy':
            preinstalled_candidates = {candidate for candidate in self._preferred_candidates if candidate.fqcn == fqcn and all((self.is_satisfied_by(requirement, candidate) for requirement in requirements))}
        try:
            coll_versions = [] if preinstalled_candidates else self._api_proxy.get_collection_versions(first_req)
        except TypeError as exc:
            if first_req.is_concrete_artifact:
                raise ValueError(f"Invalid version found for the collection '{first_req}'. {version_req}") from exc
            raise
        if first_req.is_concrete_artifact:
            for version, req_src in coll_versions:
                version_err = f"Invalid version found for the collection '{first_req}': {version} ({type(version)}). {version_req}"
                if not isinstance(version, string_types):
                    raise ValueError(version_err)
                elif version != '*':
                    try:
                        SemanticVersion(version)
                    except ValueError as ex:
                        raise ValueError(version_err) from ex
            return [Candidate(fqcn, version, _none_src_server, first_req.type, None) for version, _none_src_server in coll_versions]
        latest_matches = []
        signatures = []
        extra_signature_sources = []
        discarding_pre_releases_acceptable = any((not is_pre_release(candidate_version) for candidate_version, _src_server in coll_versions))
        all_pinned_requirement_version_numbers = {requirement.ver.lstrip('=').strip() for requirement in requirements if requirement.is_pinned} if discarding_pre_releases_acceptable else set()
        for version, src_server in coll_versions:
            tmp_candidate = Candidate(fqcn, version, src_server, 'galaxy', None)
            for requirement in requirements:
                candidate_satisfies_requirement = self.is_satisfied_by(requirement, tmp_candidate)
                if not candidate_satisfies_requirement:
                    break
                should_disregard_pre_release_candidate = is_pre_release(tmp_candidate.ver) and discarding_pre_releases_acceptable and (not (self._with_pre_releases or tmp_candidate.is_concrete_artifact or version in all_pinned_requirement_version_numbers))
                if should_disregard_pre_release_candidate:
                    break
                if not self._include_signatures:
                    continue
                extra_signature_sources.extend(requirement.signature_sources or [])
            else:
                if self._include_signatures:
                    for extra_source in extra_signature_sources:
                        signatures.append(get_signature_from_source(extra_source))
                latest_matches.append(Candidate(fqcn, version, src_server, 'galaxy', frozenset(signatures)))
        latest_matches.sort(key=lambda candidate: (SemanticVersion(candidate.ver), candidate.src), reverse=True)
        if not preinstalled_candidates:
            preinstalled_candidates = {candidate for candidate in self._preferred_candidates if candidate.fqcn == fqcn and (all((self.is_satisfied_by(requirement, candidate) for requirement in requirements)) and (not self._upgrade or all((SemanticVersion(latest.ver) <= SemanticVersion(candidate.ver) for latest in latest_matches))))}
        return list(preinstalled_candidates) + latest_matches

    def is_satisfied_by(self, requirement, candidate):
        """Whether the given requirement is satisfiable by a candidate.

        :param requirement: A requirement that produced the `candidate`.

        :param candidate: A pinned candidate supposedly matchine the \\
                          `requirement` specifier. It is guaranteed to \\
                          have been generated from the `requirement`.

        :returns: Indication whether the `candidate` is a viable \\
                  solution to the `requirement`.
        """
        if requirement.is_virtual or candidate.is_virtual or requirement.ver == '*':
            return True
        return meets_requirements(version=candidate.ver, requirements=requirement.ver)

    def get_dependencies(self, candidate):
        """Get direct dependencies of a candidate.

        :returns: A collection of requirements that `candidate` \\
                  specifies as its dependencies.
        """
        req_map = self._api_proxy.get_collection_dependencies(candidate)
        if not self._with_deps and (not candidate.is_virtual):
            return []
        return [self._make_req_from_dict({'name': dep_name, 'version': dep_req}) for dep_name, dep_req in req_map.items()]