class C3:

    @staticmethod
    def resolver(C, strict, base_mros):
        strict = strict if strict is not None else C3.STRICT_IRO
        factory = C3
        if strict:
            factory = _StrictC3
        elif C3.TRACK_BAD_IRO:
            factory = _TrackingC3
        memo = {}
        base_mros = base_mros or {}
        for base, mro in base_mros.items():
            assert base in C.__bases__
            memo[base] = _StaticMRO(base, mro)
        return factory(C, memo)
    __mro = None
    __legacy_ro = None
    direct_inconsistency = False

    def __init__(self, C, memo):
        self.leaf = C
        self.memo = memo
        kind = self.__class__
        base_resolvers = []
        for base in C.__bases__:
            if base not in memo:
                resolver = kind(base, memo)
                memo[base] = resolver
            base_resolvers.append(memo[base])
        self.base_tree = [[C]] + [memo[base].mro() for base in C.__bases__] + [list(C.__bases__)]
        self.bases_had_inconsistency = any((base.had_inconsistency for base in base_resolvers))
        if len(C.__bases__) == 1:
            self.__mro = [C] + memo[C.__bases__[0]].mro()

    @property
    def had_inconsistency(self):
        return self.direct_inconsistency or self.bases_had_inconsistency

    @property
    def legacy_ro(self):
        if self.__legacy_ro is None:
            self.__legacy_ro = tuple(_legacy_ro(self.leaf))
        return list(self.__legacy_ro)
    TRACK_BAD_IRO = _ClassBoolFromEnv()
    STRICT_IRO = _ClassBoolFromEnv()
    WARN_BAD_IRO = _ClassBoolFromEnv()
    LOG_CHANGED_IRO = _ClassBoolFromEnv()
    USE_LEGACY_IRO = _ClassBoolFromEnv()
    BAD_IROS = ()

    def _warn_iro(self):
        if not self.WARN_BAD_IRO:
            return
        import warnings
        warnings.warn("An inconsistent resolution order is being requested. (Interfaces should follow the Python class rules known as C3.) For backwards compatibility, zope.interface will allow this, making the best guess it can to produce as meaningful an order as possible. In the future this might be an error. Set the warning filter to error, or set the environment variable 'ZOPE_INTERFACE_TRACK_BAD_IRO' to '1' and examine ro.C3.BAD_IROS to debug, or set 'ZOPE_INTERFACE_STRICT_IRO' to raise exceptions.", InconsistentResolutionOrderWarning)

    @staticmethod
    def _can_choose_base(base, base_tree_remaining):
        for bases in base_tree_remaining:
            if not bases or bases[0] is base:
                continue
            for b in bases:
                if b is base:
                    return False
        return True

    @staticmethod
    def _nonempty_bases_ignoring(base_tree, ignoring):
        return list(filter(None, [[b for b in bases if b is not ignoring] for bases in base_tree]))

    def _choose_next_base(self, base_tree_remaining):
        """
        Return the next base.

        The return value will either fit the C3 constraints or be our best
        guess about what to do. If we cannot guess, this may raise an exception.
        """
        base = self._find_next_C3_base(base_tree_remaining)
        if base is not None:
            return base
        return self._guess_next_base(base_tree_remaining)

    def _find_next_C3_base(self, base_tree_remaining):
        """
        Return the next base that fits the constraints, or ``None`` if there isn't one.
        """
        for bases in base_tree_remaining:
            base = bases[0]
            if self._can_choose_base(base, base_tree_remaining):
                return base
        return None

    class _UseLegacyRO(Exception):
        pass

    def _guess_next_base(self, base_tree_remaining):
        self._warn_iro()
        self.direct_inconsistency = InconsistentResolutionOrderError(self, base_tree_remaining)
        raise self._UseLegacyRO

    def _merge(self):
        result = self.__mro = []
        base_tree_remaining = self.base_tree
        base = None
        while 1:
            base_tree_remaining = self._nonempty_bases_ignoring(base_tree_remaining, base)
            if not base_tree_remaining:
                return result
            try:
                base = self._choose_next_base(base_tree_remaining)
            except self._UseLegacyRO:
                self.__mro = self.legacy_ro
                return self.legacy_ro
            result.append(base)

    def mro(self):
        if self.__mro is None:
            self.__mro = tuple(self._merge())
        return list(self.__mro)