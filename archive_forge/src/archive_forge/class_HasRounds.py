from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
class HasRounds(GenericHandler):
    """mixin for validating rounds parameter

    This :class:`GenericHandler` mixin adds a ``rounds`` keyword to the class
    constuctor; any value provided is passed through the :meth:`_norm_rounds`
    method, which takes care of validating the number of rounds.

    :param rounds: optional number of rounds hash should use

    Class Attributes
    ================
    In order for :meth:`!_norm_rounds` to do its job, the following
    attributes must be provided by the handler subclass:

    .. attribute:: min_rounds

        The minimum number of rounds allowed. A :exc:`ValueError` will be
        thrown if the rounds value is too small. Defaults to ``0``.

    .. attribute:: max_rounds

        The maximum number of rounds allowed. A :exc:`ValueError` will be
        thrown if the rounds value is larger than this. Defaults to ``None``
        which indicates no limit to the rounds value.

    .. attribute:: default_rounds

        If no rounds value is provided to constructor, this value will be used.
        If this is not specified, a rounds value *must* be specified by the
        application.

    .. attribute:: rounds_cost

        [required]
        The ``rounds`` parameter typically encodes a cpu-time cost
        for calculating a hash. This should be set to ``"linear"``
        (the default) or ``"log2"``, depending on how the rounds value relates
        to the actual amount of time that will be required.

    Class Methods
    =============
    .. todo:: document using() and needs_update() options

    Instance Attributes
    ===================
    .. attribute:: rounds

        This instance attribute will be filled in with the rounds value provided
        to the constructor (as adapted by :meth:`_norm_rounds`)

    Subclassable Methods
    ====================
    .. automethod:: _norm_rounds
    """
    min_rounds = 0
    max_rounds = None
    rounds_cost = 'linear'
    using_rounds_kwds = ('min_desired_rounds', 'max_desired_rounds', 'min_rounds', 'max_rounds', 'default_rounds', 'vary_rounds')
    min_desired_rounds = None
    max_desired_rounds = None
    default_rounds = None
    vary_rounds = None
    rounds = None

    @classmethod
    def using(cls, min_desired_rounds=None, max_desired_rounds=None, default_rounds=None, vary_rounds=None, min_rounds=None, max_rounds=None, rounds=None, **kwds):
        if min_rounds is not None:
            if min_desired_rounds is not None:
                raise TypeError("'min_rounds' and 'min_desired_rounds' aliases are mutually exclusive")
            min_desired_rounds = min_rounds
        if max_rounds is not None:
            if max_desired_rounds is not None:
                raise TypeError("'max_rounds' and 'max_desired_rounds' aliases are mutually exclusive")
            max_desired_rounds = max_rounds
        if rounds is not None:
            if min_desired_rounds is None:
                min_desired_rounds = rounds
            if max_desired_rounds is None:
                max_desired_rounds = rounds
            if default_rounds is None:
                default_rounds = rounds
        subcls = super(HasRounds, cls).using(**kwds)
        relaxed = kwds.get('relaxed')
        if min_desired_rounds is None:
            explicit_min_rounds = False
            min_desired_rounds = cls.min_desired_rounds
        else:
            explicit_min_rounds = True
            if isinstance(min_desired_rounds, native_string_types):
                min_desired_rounds = int(min_desired_rounds)
            subcls.min_desired_rounds = subcls._norm_rounds(min_desired_rounds, param='min_desired_rounds', relaxed=relaxed)
        if max_desired_rounds is None:
            max_desired_rounds = cls.max_desired_rounds
        else:
            if isinstance(max_desired_rounds, native_string_types):
                max_desired_rounds = int(max_desired_rounds)
            if min_desired_rounds and max_desired_rounds < min_desired_rounds:
                msg = '%s: max_desired_rounds (%r) below min_desired_rounds (%r)' % (subcls.name, max_desired_rounds, min_desired_rounds)
                if explicit_min_rounds:
                    raise ValueError(msg)
                else:
                    warn(msg, PasslibConfigWarning)
                    max_desired_rounds = min_desired_rounds
            subcls.max_desired_rounds = subcls._norm_rounds(max_desired_rounds, param='max_desired_rounds', relaxed=relaxed)
        if default_rounds is not None:
            if isinstance(default_rounds, native_string_types):
                default_rounds = int(default_rounds)
            if min_desired_rounds and default_rounds < min_desired_rounds:
                raise ValueError('%s: default_rounds (%r) below min_desired_rounds (%r)' % (subcls.name, default_rounds, min_desired_rounds))
            elif max_desired_rounds and default_rounds > max_desired_rounds:
                raise ValueError('%s: default_rounds (%r) above max_desired_rounds (%r)' % (subcls.name, default_rounds, max_desired_rounds))
            subcls.default_rounds = subcls._norm_rounds(default_rounds, param='default_rounds', relaxed=relaxed)
        if subcls.default_rounds is not None:
            subcls.default_rounds = subcls._clip_to_desired_rounds(subcls.default_rounds)
        if vary_rounds is not None:
            if isinstance(vary_rounds, native_string_types):
                if vary_rounds.endswith('%'):
                    vary_rounds = float(vary_rounds[:-1]) * 0.01
                elif '.' in vary_rounds:
                    vary_rounds = float(vary_rounds)
                else:
                    vary_rounds = int(vary_rounds)
            if vary_rounds < 0:
                raise ValueError('%s: vary_rounds (%r) below 0' % (subcls.name, vary_rounds))
            elif isinstance(vary_rounds, float):
                if vary_rounds > 1:
                    raise ValueError('%s: vary_rounds (%r) above 1.0' % (subcls.name, vary_rounds))
            elif not isinstance(vary_rounds, int):
                raise TypeError('vary_rounds must be int or float')
            if vary_rounds:
                warn("The 'vary_rounds' option is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0", PasslibConfigWarning)
            subcls.vary_rounds = vary_rounds
        return subcls

    @classmethod
    def _clip_to_desired_rounds(cls, rounds):
        """
        helper for :meth:`_generate_rounds` --
        clips rounds value to desired min/max set by class (if any)
        """
        mnd = cls.min_desired_rounds or 0
        if rounds < mnd:
            return mnd
        mxd = cls.max_desired_rounds
        if mxd and rounds > mxd:
            return mxd
        return rounds

    @classmethod
    def _calc_vary_rounds_range(cls, default_rounds):
        """
        helper for :meth:`_generate_rounds` --
        returns range for vary rounds generation.

        :returns:
            (lower, upper) limits suitable for random.randint()
        """
        assert default_rounds
        vary_rounds = cls.vary_rounds

        def linear_to_native(value, upper):
            return value
        if isinstance(vary_rounds, float):
            assert 0 <= vary_rounds <= 1
            if cls.rounds_cost == 'log2':
                default_rounds = 1 << default_rounds

                def linear_to_native(value, upper):
                    if value <= 0:
                        return 0
                    elif upper:
                        return int(math.log(value, 2))
                    else:
                        return int(math.ceil(math.log(value, 2)))
            vary_rounds = int(default_rounds * vary_rounds)
        assert vary_rounds >= 0 and isinstance(vary_rounds, int_types)
        lower = linear_to_native(default_rounds - vary_rounds, False)
        upper = linear_to_native(default_rounds + vary_rounds, True)
        return (cls._clip_to_desired_rounds(lower), cls._clip_to_desired_rounds(upper))

    def __init__(self, rounds=None, **kwds):
        super(HasRounds, self).__init__(**kwds)
        if rounds is not None:
            rounds = self._parse_rounds(rounds)
        elif self.use_defaults:
            rounds = self._generate_rounds()
            assert self._norm_rounds(rounds) == rounds, 'generated invalid rounds: %r' % (rounds,)
        else:
            raise TypeError('no rounds specified')
        self.rounds = rounds

    def _parse_rounds(self, rounds):
        return self._norm_rounds(rounds)

    @classmethod
    def _norm_rounds(cls, rounds, relaxed=False, param='rounds'):
        """
        helper for normalizing rounds value.

        :arg rounds:
            an integer cost parameter.

        :param relaxed:
            if ``True`` (the default), issues PasslibHashWarning is rounds are outside allowed range.
            if ``False``, raises a ValueError instead.

        :param param:
            optional name of parameter to insert into error/warning messages.

        :raises TypeError:
            * if ``use_defaults=False`` and no rounds is specified
            * if rounds is not an integer.

        :raises ValueError:

            * if rounds is ``None`` and class does not specify a value for
              :attr:`default_rounds`.
            * if ``relaxed=False`` and rounds is outside bounds of
              :attr:`min_rounds` and :attr:`max_rounds` (if ``relaxed=True``,
              the rounds value will be clamped, and a warning issued).

        :returns:
            normalized rounds value
        """
        return norm_integer(cls, rounds, cls.min_rounds, cls.max_rounds, param=param, relaxed=relaxed)

    @classmethod
    def _generate_rounds(cls):
        """
        internal helper for :meth:`_norm_rounds` --
        returns default rounds value, incorporating vary_rounds,
        and any other limitations hash may place on rounds parameter.
        """
        rounds = cls.default_rounds
        if rounds is None:
            raise TypeError('%s rounds value must be specified explicitly' % (cls.name,))
        if cls.vary_rounds:
            lower, upper = cls._calc_vary_rounds_range(rounds)
            assert lower <= rounds <= upper
            if lower < upper:
                rounds = rng.randint(lower, upper)
        return rounds

    def _calc_needs_update(self, **kwds):
        """
        mark hash as needing update if rounds is outside desired bounds.
        """
        min_desired_rounds = self.min_desired_rounds
        if min_desired_rounds and self.rounds < min_desired_rounds:
            return True
        max_desired_rounds = self.max_desired_rounds
        if max_desired_rounds and self.rounds > max_desired_rounds:
            return True
        return super(HasRounds, self)._calc_needs_update(**kwds)

    @classmethod
    def bitsize(cls, rounds=None, vary_rounds=0.1, **kwds):
        """[experimental method] return info about bitsizes of hash"""
        info = super(HasRounds, cls).bitsize(**kwds)
        if cls.rounds_cost != 'log2':
            import math
            if rounds is None:
                rounds = cls.default_rounds
            info['rounds'] = max(0, int(1 + math.log(rounds * vary_rounds, 2)))
        return info