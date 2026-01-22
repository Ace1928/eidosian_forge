from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def apply_single_tag_set(tag_set: TagSet, selection: Selection) -> Selection:
    """All servers matching one tag set.

    A tag set is a dict. A server matches if its tags are a superset:
    A server tagged {'a': '1', 'b': '2'} matches the tag set {'a': '1'}.

    The empty tag set {} matches any server.
    """

    def tags_match(server_tags: Mapping[str, Any]) -> bool:
        for key, value in tag_set.items():
            if key not in server_tags or server_tags[key] != value:
                return False
        return True
    return selection.with_server_descriptions([s for s in selection.server_descriptions if tags_match(s.tags)])