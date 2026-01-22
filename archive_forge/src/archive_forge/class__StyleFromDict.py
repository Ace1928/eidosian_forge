from .base import Style, DEFAULT_ATTRS, ANSI_COLOR_NAMES
from .defaults import DEFAULT_STYLE_EXTENSIONS
from .utils import merge_attrs, split_token_in_parts
from six.moves import range
class _StyleFromDict(Style):
    """
    Turn a dictionary that maps `Token` to `Attrs` into a style class.

    :param token_to_attrs: Dictionary that maps `Token` to `Attrs`.
    """

    def __init__(self, token_to_attrs):
        self.token_to_attrs = token_to_attrs

    def get_attrs_for_token(self, token):
        list_of_attrs = []
        for token in split_token_in_parts(token):
            list_of_attrs.append(self.token_to_attrs.get(token, DEFAULT_ATTRS))
        return merge_attrs(list_of_attrs)

    def invalidation_hash(self):
        return id(self.token_to_attrs)