from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _parse_and_normalize_offers(cls, offers):
    """
        Throw out any offers that do not match the media range ABNF.

        :return: A list of offers split into the format ``[offer_index,
                 parsed_offer]``.

        """
    parsed_offers = []
    for index, offer in enumerate(offers):
        try:
            parsed_offer = cls.parse_offer(offer)
        except ValueError:
            continue
        parsed_offers.append([index, parsed_offer])
    return parsed_offers