from collections import namedtuple
import re
import textwrap
import warnings
def acceptable_offers(self, offers):
    """
        Return the offers that are acceptable according to the header.

        :param offers: ``iterable`` of ``str``s, where each ``str`` is a
                       content-coding or the string ``identity`` (the token
                       used to represent "no encoding")
        :return: When the header is invalid, or there is no ``Accept-Encoding``
                 header in the request, all `offers` are considered acceptable,
                 so this method returns a list of (content-coding or
                 "identity", qvalue) tuples where each offer in `offers` is
                 paired with the qvalue of 1.0, in the same order as in
                 `offers`.
        """
    return [(offer, 1.0) for offer in offers]