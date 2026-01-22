from collections import namedtuple
import re
import textwrap
import warnings
class _AcceptLanguageInvalidOrNoHeader(AcceptLanguage):
    """
    Represent when an ``Accept-Language`` header is invalid or not in request.

    This is the base class for the behaviour that
    :class:`.AcceptLanguageInvalidHeader` and :class:`.AcceptLanguageNoHeader`
    have in common.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``Accept-Language`` header has an invalid value. This implementation
    disregards the header when the header is invalid, so
    :class:`.AcceptLanguageInvalidHeader` and :class:`.AcceptLanguageNoHeader`
    have much behaviour in common.
    """

    def __nonzero__(self):
        """
        Return whether ``self`` represents a valid ``Accept-Language`` header.

        Return ``True`` if ``self`` represents a valid header, and ``False`` if
        it represents an invalid header, or the header not being in the
        request.

        For this class, it always returns ``False``.
        """
        return False
    __bool__ = __nonzero__

    def __contains__(self, offer):
        """
        Return ``bool`` indicating whether `offer` is acceptable.

        .. warning::

           The behavior of ``.__contains__`` for the ``AcceptLanguage`` classes
           is currently being maintained for backward compatibility, but it
           will change in the future to better conform to the RFC.

        :param offer: (``str``) language tag offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        For this class, either there is no ``Accept-Language`` header in the
        request, or the header is invalid, so any language tag is acceptable,
        and this always returns ``True``.
        """
        warnings.warn('The behavior of .__contains__ for the AcceptLanguage classes is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        return True

    def __iter__(self):
        """
        Return all the ranges with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the language ranges in the header with non-0
                 qvalues, in descending order of qvalue. If two ranges have the
                 same qvalue, they are returned in the order of their positions
                 in the header, from left to right.

        For this class, either there is no ``Accept-Language`` header in the
        request, or the header is invalid, so there are no language ranges, and
        this always returns an empty iterator.
        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        return iter(())

    def basic_filtering(self, language_tags):
        """
        Return the tags that match the header, using Basic Filtering.

        :param language_tags: (``iterable``) language tags
        :return: A list of tuples of the form (language tag, qvalue), in
                 descending order of preference.

        When the header is invalid and when the header is not in the request,
        there are no matches, so this method always returns an empty list.
        """
        return []

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of language tag `offers`.

        This is the ``.best_match()`` method for when the header is invalid or
        not found in the request, corresponding to
        :meth:`AcceptLanguageValidHeader.best_match`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptLanguageValidHeader.best_match`).

        When the header is invalid, or there is no `Accept-Language` header in
        the request, any of the language tags in `offers` are considered
        acceptable, so the best match is the tag in `offers` with the highest
        server quality value (if the server quality value is not supplied, it
        is 1).

        If more than one language tags in `offers` have the same highest server
        quality value, then the one that shows up first in `offers` is the best
        match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` language
                         tag, or a (language tag, server quality value)
                         ``tuple`` or ``list``. (The two may be mixed in the
                         iterable.)

        :param default_match: (optional, any type) the value to be returned if
                              `offers` is empty.

        :return: (``str``, or the type of `default_match`)

                 | The language tag that has the highest server quality value.
                   If `offers` is empty, the value of `default_match` is
                   returned.
        """
        warnings.warn('The behavior of .best_match for the AcceptLanguage classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior is not specified in (and currently does not conform to) RFC 7231.', DeprecationWarning)
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

    def lookup(self, language_tags=None, default_range=None, default_tag=None, default=None):
        """
        Return the language tag that best matches the header, using Lookup.

        When the header is invalid, or there is no ``Accept-Language`` header
        in the request, all language tags are considered acceptable, so it is
        as if the header is '*'. As specified for the Lookup matching scheme in
        :rfc:`RFC 4647, section 3.4 <4647#section-3.4>`, when the header is
        '*', the default value is to be computed and returned. So this method
        will ignore the `language_tags` and `default_range` arguments, and
        proceed to `default_tag`, then `default`.

        :param language_tags: (optional, any type)

                              | This argument is ignored, and is only used as a
                                placeholder so that the method signature
                                corresponds to that of
                                :meth:`AcceptLanguageValidHeader.lookup`.

        :param default_range: (optional, any type)

                              | This argument is ignored, and is only used as a
                                placeholder so that the method signature
                                corresponds to that of
                                :meth:`AcceptLanguageValidHeader.lookup`.

        :param default_tag: (optional, ``None`` or ``str``)

                            | At least one of `default_tag` or `default` must
                              be supplied as an argument to the method, to
                              define the defaulting behaviour.

                            | If this argument is not ``None``, then it is
                              returned.

                            | This parameter corresponds to "return a
                              particular language tag designated for the
                              operation", one of the examples of "defaulting
                              behavior" described in :rfc:`RFC 4647, section
                              3.4.1 <4647#section-3.4.1>`.

        :param default: (optional, ``None`` or any type, including a callable)

                        | At least one of `default_tag` or `default` must be
                          supplied as an argument to the method, to define the
                          defaulting behaviour.

                        | If `default_tag` is ``None``, then Lookup will next
                          examine the `default` argument.

                        | If `default` is a callable, it will be called, and
                          the callable's return value will be returned.

                        | If `default` is not a callable, the value itself will
                          be returned.

                        | This parameter corresponds to the "defaulting
                          behavior" described in :rfc:`RFC 4647, section 3.4.1
                          <4647#section-3.4.1>`

        :return: (``str``, or any type)

                 | the return value from `default_tag` or `default`.
        """
        if default_tag is None and default is None:
            raise TypeError('`default_tag` and `default` arguments cannot both be None.')
        if default_tag is not None:
            return default_tag
        try:
            return default()
        except TypeError:
            return default

    def quality(self, offer):
        """
        Return quality value of given offer, or ``None`` if there is no match.

        This is the ``.quality()`` method for when the header is invalid or not
        found in the request, corresponding to
        :meth:`AcceptLanguageValidHeader.quality`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptLanguageValidHeader.quality`).

        :param offer: (``str``) language tag offer
        :return: (``float``) ``1.0``.

        When the ``Accept-Language`` header is invalid or not in the request,
        all offers are equally acceptable, so 1.0 is always returned.
        """
        warnings.warn('The behavior of .quality for the AcceptLanguage classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior is not specified in (and currently does not conform to) RFC 7231.', DeprecationWarning)
        return 1.0