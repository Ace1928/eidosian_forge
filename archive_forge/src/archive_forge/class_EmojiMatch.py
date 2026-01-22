from typing import NamedTuple, Dict, Union, Iterator, Any
from emoji import unicode_codes
class EmojiMatch:
    """
    Represents a match of a "recommended for general interchange" (RGI)
    emoji in a string.
    """
    __slots__ = ('emoji', 'start', 'end', 'data')

    def __init__(self, emoji: str, start: int, end: int, data: Union[dict, None]):
        self.emoji = emoji
        'The emoji substring'
        self.start = start
        'The start index of the match in the string'
        self.end = end
        'The end index of the match in the string'
        self.data = data
        'The entry from :data:`EMOJI_DATA` for this emoji or ``None`` if the emoji is non-RGI'

    def data_copy(self) -> Dict[str, Any]:
        """
        Returns a copy of the data from :data:`EMOJI_DATA` for this match
        with the additional keys ``match_start`` and ``match_end``.
        """
        if self.data:
            emj_data = self.data.copy()
            emj_data['match_start'] = self.start
            emj_data['match_end'] = self.end
            return emj_data
        else:
            return {'match_start': self.start, 'match_end': self.end}

    def is_zwj(self) -> bool:
        """
        Checks if this is a ZWJ-emoji.

        :returns: True if this is a ZWJ-emoji, False otherwise
        """
        return _ZWJ in self.emoji

    def split(self) -> Union['EmojiMatchZWJ', 'EmojiMatch']:
        """
        Splits a ZWJ-emoji into its constituents.

        :returns: An :class:`EmojiMatchZWJ` containing the "sub-emoji" if this is a ZWJ-emoji, otherwise self
        """
        if self.is_zwj():
            return EmojiMatchZWJ(self)
        else:
            return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.emoji}, {self.start}:{self.end})'