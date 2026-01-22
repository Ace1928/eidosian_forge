from ._impl import Mismatch
def Always():
    """Always match.

    That is::

        self.assertThat(x, Always())

    Will always match and never fail, no matter what ``x`` is. Most useful when
    passed to other higher-order matchers (e.g.
    :py:class:`~testtools.matchers.MatchesListwise`).
    """
    return _Always()