from collections import namedtuple
import re
import textwrap
import warnings
def _old_match(self, mask, item):
    """
        Return whether a language tag matches a language range.

        .. warning::

           This is maintained for backward compatibility, and will be
           deprecated in the future.

        This method was WebOb's old criterion for deciding whether a language
        tag matches a language range, used in

        - :meth:`AcceptLanguageValidHeader.__contains__`
        - :meth:`AcceptLanguageValidHeader.best_match`
        - :meth:`AcceptLanguageValidHeader.quality`

        It does not conform to :rfc:`RFC 7231, section 5.3.5
        <7231#section-5.3.5>`, or any of the matching schemes suggested there.

        :param mask: (``str``)

                     | language range

        :param item: (``str``)

                     | language tag. Subtags in language tags are separated by
                       ``-`` (hyphen). If there are underscores (``_``) in this
                       argument, they will be converted to hyphens before
                       checking the match.

        :return: (``bool``) whether the tag in `item` matches the range in
                 `mask`.

        `mask` and `item` are a match if:

        - ``mask == *``.
        - ``mask == item``.
        - If the first subtag of `item` equals `mask`, or if the first subtag
          of `mask` equals `item`.
          This means that::

              >>> instance._old_match(mask='en-gb', item='en')
              True
              >>> instance._old_match(mask='en', item='en-gb')
              True

          Which is different from any of the matching schemes suggested in
          :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>`, in that none of
          those schemes match both more *and* less specific tags.

          However, this method appears to be only designed for language tags
          and ranges with at most two subtags. So with an `item`/language tag
          with more than two subtags like ``zh-Hans-CN``::

              >>> instance._old_match(mask='zh', item='zh-Hans-CN')
              True
              >>> instance._old_match(mask='zh-Hans', item='zh-Hans-CN')
              False

          From commit history, this does not appear to have been from a
          decision to match only the first subtag, but rather because only
          language ranges and tags with at most two subtags were expected.
        """
    item = item.replace('_', '-').lower()
    mask = mask.lower()
    return mask == '*' or item == mask or item.split('-')[0] == mask or (item == mask.split('-')[0])