from collections import namedtuple
import re
import textwrap
import warnings
class _AcceptInvalidOrNoHeader(Accept):
    """
    Represent when an ``Accept`` header is invalid or not in request.

    This is the base class for the behaviour that :class:`.AcceptInvalidHeader`
    and :class:`.AcceptNoHeader` have in common.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``Accept`` header has an invalid value. This implementation disregards the
    header when the header is invalid, so :class:`.AcceptInvalidHeader` and
    :class:`.AcceptNoHeader` have much behaviour in common.
    """

    def __bool__(self):
        """
        Return whether ``self`` represents a valid ``Accept`` header.

        Return ``True`` if ``self`` represents a valid header, and ``False`` if
        it represents an invalid header, or the header not being in the
        request.

        For this class, it always returns ``False``.
        """
        return False
    __nonzero__ = __bool__

    def __contains__(self, offer):
        """
        Return ``bool`` indicating whether `offer` is acceptable.

        .. warning::

           The behavior of ``.__contains__`` for the ``Accept`` classes is
           currently being maintained for backward compatibility, but it will
           change in the future to better conform to the RFC.

        :param offer: (``str``) media type offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        For this class, either there is no ``Accept`` header in the request, or
        the header is invalid, so any media type is acceptable, and this always
        returns ``True``.
        """
        warnings.warn('The behavior of .__contains__ for the Accept classes is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        return True

    def __iter__(self):
        """
        Return all the ranges with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the media ranges in the header with non-0
                 qvalues, in descending order of qvalue. If two ranges have the
                 same qvalue, they are returned in the order of their positions
                 in the header, from left to right.

        When there is no ``Accept`` header in the request or the header is
        invalid, there are no media ranges, so this always returns an empty
        iterator.
        """
        warnings.warn('The behavior of AcceptValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        return iter(())

    def accept_html(self):
        """
        Return ``True`` if any HTML-like type is accepted.

        The HTML-like types are 'text/html', 'application/xhtml+xml',
        'application/xml' and 'text/xml'.

        When the header is invalid, or there is no `Accept` header in the
        request, all `offers` are considered acceptable, so this always returns
        ``True``.
        """
        return bool(self.acceptable_offers(offers=['text/html', 'application/xhtml+xml', 'application/xml', 'text/xml']))
    accepts_html = property(fget=accept_html, doc=accept_html.__doc__)

    def acceptable_offers(self, offers):
        """
        Return the offers that are acceptable according to the header.

        Any offers that cannot be parsed via
        :meth:`.Accept.parse_offer` will be ignored.

        :param offers: ``iterable`` of ``str`` media types (media types can
                       include media type parameters)
        :return: When the header is invalid, or there is no ``Accept`` header
                 in the request, all `offers` are considered acceptable, so
                 this method returns a list of (media type, qvalue) tuples
                 where each offer in `offers` is paired with the qvalue of 1.0,
                 in the same order as in `offers`.
        """
        return [(offers[offer_index], 1.0) for offer_index, _ in self._parse_and_normalize_offers(offers)]

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of language tag `offers`.

        This is the ``.best_match()`` method for when the header is invalid or
        not found in the request, corresponding to
        :meth:`AcceptValidHeader.best_match`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptValidHeader.best_match`).

        When the header is invalid, or there is no `Accept` header in the
        request, all `offers` are considered acceptable, so the best match is
        the media type in `offers` with the highest server quality value (if
        the server quality value is not supplied for a media type, it is 1).

        If more than one media type in `offers` have the same highest server
        quality value, then the one that shows up first in `offers` is the best
        match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` media type,
                         or a (media type, server quality value) ``tuple`` or
                         ``list``. (The two may be mixed in the iterable.)

        :param default_match: (optional, any type) the value to be returned if
                              `offers` is empty.

        :return: (``str``, or the type of `default_match`)

                 | The offer that has the highest server quality value.  If
                   `offers` is empty, the value of `default_match` is returned.
        """
        warnings.warn('The behavior of .best_match for the Accept classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior is not specified in (and currently does not conform to) RFC 7231.', DeprecationWarning)
        best_quality = -1
        best_offer = default_match
        for offer in offers:
            if isinstance(offer, (list, tuple)):
                offer, quality = offer
            else:
                quality = 1
            if quality > best_quality:
                best_offer = offer
                best_quality = quality
        return best_offer

    def quality(self, offer):
        """
        Return quality value of given offer, or ``None`` if there is no match.

        This is the ``.quality()`` method for when the header is invalid or not
        found in the request, corresponding to
        :meth:`AcceptValidHeader.quality`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptValidHeader.quality`).

        :param offer: (``str``) media type offer
        :return: (``float``) ``1.0``.

        When the ``Accept`` header is invalid or not in the request, all offers
        are equally acceptable, so 1.0 is always returned.
        """
        warnings.warn('The behavior of .quality for the Accept classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior does not conform toRFC 7231.', DeprecationWarning)
        return 1.0