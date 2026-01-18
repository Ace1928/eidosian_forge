def tuple2str(tagged_token, sep='/'):
    """
    Given the tuple representation of a tagged token, return the
    corresponding string representation.  This representation is
    formed by concatenating the token's word string, followed by the
    separator, followed by the token's tag.  (If the tag is None,
    then just return the bare word string.)

        >>> from nltk.tag.util import tuple2str
        >>> tagged_token = ('fly', 'NN')
        >>> tuple2str(tagged_token)
        'fly/NN'

    :type tagged_token: tuple(str, str)
    :param tagged_token: The tuple representation of a tagged token.
    :type sep: str
    :param sep: The separator string used to separate word strings
        from tags.
    """
    word, tag = tagged_token
    if tag is None:
        return word
    else:
        assert sep not in tag, 'tag may not contain sep!'
        return f'{word}{sep}{tag}'