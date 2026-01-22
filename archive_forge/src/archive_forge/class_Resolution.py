import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
class Resolution(object):
    """Stateful resolution object.

    This is designed as a one-off object that holds information to kick start
    the resolution process, and holds the results afterwards.
    """

    def __init__(self, provider, reporter):
        self._p = provider
        self._r = reporter
        self._states = []

    @property
    def state(self):
        try:
            return self._states[-1]
        except IndexError:
            raise AttributeError('state')

    def _push_new_state(self):
        """Push a new state into history.

        This new state will be used to hold resolution results of the next
        coming round.
        """
        base = self._states[-1]
        state = State(mapping=base.mapping.copy(), criteria=base.criteria.copy(), backtrack_causes=base.backtrack_causes[:])
        self._states.append(state)

    def _add_to_criteria(self, criteria, requirement, parent):
        self._r.adding_requirement(requirement=requirement, parent=parent)
        identifier = self._p.identify(requirement_or_candidate=requirement)
        criterion = criteria.get(identifier)
        if criterion:
            incompatibilities = list(criterion.incompatibilities)
        else:
            incompatibilities = []
        matches = self._p.find_matches(identifier=identifier, requirements=IteratorMapping(criteria, operator.methodcaller('iter_requirement'), {identifier: [requirement]}), incompatibilities=IteratorMapping(criteria, operator.attrgetter('incompatibilities'), {identifier: incompatibilities}))
        if criterion:
            information = list(criterion.information)
            information.append(RequirementInformation(requirement, parent))
        else:
            information = [RequirementInformation(requirement, parent)]
        criterion = Criterion(candidates=build_iter_view(matches), information=information, incompatibilities=incompatibilities)
        if not criterion.candidates:
            raise RequirementsConflicted(criterion)
        criteria[identifier] = criterion

    def _remove_information_from_criteria(self, criteria, parents):
        """Remove information from parents of criteria.

        Concretely, removes all values from each criterion's ``information``
        field that have one of ``parents`` as provider of the requirement.

        :param criteria: The criteria to update.
        :param parents: Identifiers for which to remove information from all criteria.
        """
        if not parents:
            return
        for key, criterion in criteria.items():
            criteria[key] = Criterion(criterion.candidates, [information for information in criterion.information if information.parent is None or self._p.identify(information.parent) not in parents], criterion.incompatibilities)

    def _get_preference(self, name):
        return self._p.get_preference(identifier=name, resolutions=self.state.mapping, candidates=IteratorMapping(self.state.criteria, operator.attrgetter('candidates')), information=IteratorMapping(self.state.criteria, operator.attrgetter('information')), backtrack_causes=self.state.backtrack_causes)

    def _is_current_pin_satisfying(self, name, criterion):
        try:
            current_pin = self.state.mapping[name]
        except KeyError:
            return False
        return all((self._p.is_satisfied_by(requirement=r, candidate=current_pin) for r in criterion.iter_requirement()))

    def _get_updated_criteria(self, candidate):
        criteria = self.state.criteria.copy()
        for requirement in self._p.get_dependencies(candidate=candidate):
            self._add_to_criteria(criteria, requirement, parent=candidate)
        return criteria

    def _attempt_to_pin_criterion(self, name):
        criterion = self.state.criteria[name]
        causes = []
        for candidate in criterion.candidates:
            try:
                criteria = self._get_updated_criteria(candidate)
            except RequirementsConflicted as e:
                self._r.rejecting_candidate(e.criterion, candidate)
                causes.append(e.criterion)
                continue
            satisfied = all((self._p.is_satisfied_by(requirement=r, candidate=candidate) for r in criterion.iter_requirement()))
            if not satisfied:
                raise InconsistentCandidate(candidate, criterion)
            self._r.pinning(candidate=candidate)
            self.state.criteria.update(criteria)
            self.state.mapping.pop(name, None)
            self.state.mapping[name] = candidate
            return []
        return causes

    def _backjump(self, causes):
        """Perform backjumping.

        When we enter here, the stack is like this::

            [ state Z ]
            [ state Y ]
            [ state X ]
            .... earlier states are irrelevant.

        1. No pins worked for Z, so it does not have a pin.
        2. We want to reset state Y to unpinned, and pin another candidate.
        3. State X holds what state Y was before the pin, but does not
           have the incompatibility information gathered in state Y.

        Each iteration of the loop will:

        1.  Identify Z. The incompatibility is not always caused by the latest
            state. For example, given three requirements A, B and C, with
            dependencies A1, B1 and C1, where A1 and B1 are incompatible: the
            last state might be related to C, so we want to discard the
            previous state.
        2.  Discard Z.
        3.  Discard Y but remember its incompatibility information gathered
            previously, and the failure we're dealing with right now.
        4.  Push a new state Y' based on X, and apply the incompatibility
            information from Y to Y'.
        5a. If this causes Y' to conflict, we need to backtrack again. Make Y'
            the new Z and go back to step 2.
        5b. If the incompatibilities apply cleanly, end backtracking.
        """
        incompatible_reqs = itertools.chain((c.parent for c in causes if c.parent is not None), (c.requirement for c in causes))
        incompatible_deps = {self._p.identify(r) for r in incompatible_reqs}
        while len(self._states) >= 3:
            del self._states[-1]
            incompatible_state = False
            while not incompatible_state:
                try:
                    broken_state = self._states.pop()
                    name, candidate = broken_state.mapping.popitem()
                except (IndexError, KeyError):
                    raise ResolutionImpossible(causes)
                current_dependencies = {self._p.identify(d) for d in self._p.get_dependencies(candidate)}
                incompatible_state = not current_dependencies.isdisjoint(incompatible_deps)
            incompatibilities_from_broken = [(k, list(v.incompatibilities)) for k, v in broken_state.criteria.items()]
            incompatibilities_from_broken.append((name, [candidate]))

            def _patch_criteria():
                for k, incompatibilities in incompatibilities_from_broken:
                    if not incompatibilities:
                        continue
                    try:
                        criterion = self.state.criteria[k]
                    except KeyError:
                        continue
                    matches = self._p.find_matches(identifier=k, requirements=IteratorMapping(self.state.criteria, operator.methodcaller('iter_requirement')), incompatibilities=IteratorMapping(self.state.criteria, operator.attrgetter('incompatibilities'), {k: incompatibilities}))
                    candidates = build_iter_view(matches)
                    if not candidates:
                        return False
                    incompatibilities.extend(criterion.incompatibilities)
                    self.state.criteria[k] = Criterion(candidates=candidates, information=list(criterion.information), incompatibilities=incompatibilities)
                return True
            self._push_new_state()
            success = _patch_criteria()
            if success:
                return True
        return False

    def resolve(self, requirements, max_rounds):
        if self._states:
            raise RuntimeError('already resolved')
        self._r.starting()
        self._states = [State(mapping=collections.OrderedDict(), criteria={}, backtrack_causes=[])]
        for r in requirements:
            try:
                self._add_to_criteria(self.state.criteria, r, parent=None)
            except RequirementsConflicted as e:
                raise ResolutionImpossible(e.criterion.information)
        self._push_new_state()
        for round_index in range(max_rounds):
            self._r.starting_round(index=round_index)
            unsatisfied_names = [key for key, criterion in self.state.criteria.items() if not self._is_current_pin_satisfying(key, criterion)]
            if not unsatisfied_names:
                self._r.ending(state=self.state)
                return self.state
            satisfied_names = set(self.state.criteria.keys()) - set(unsatisfied_names)
            name = min(unsatisfied_names, key=self._get_preference)
            failure_causes = self._attempt_to_pin_criterion(name)
            if failure_causes:
                causes = [i for c in failure_causes for i in c.information]
                self._r.resolving_conflicts(causes=causes)
                success = self._backjump(causes)
                self.state.backtrack_causes[:] = causes
                if not success:
                    raise ResolutionImpossible(self.state.backtrack_causes)
            else:
                newly_unsatisfied_names = {key for key, criterion in self.state.criteria.items() if key in satisfied_names and (not self._is_current_pin_satisfying(key, criterion))}
                self._remove_information_from_criteria(self.state.criteria, newly_unsatisfied_names)
                self._push_new_state()
            self._r.ending_round(index=round_index, state=self.state)
        raise ResolutionTooDeep(max_rounds)