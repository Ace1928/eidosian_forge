from __future__ import annotations
from copy import deepcopy
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.dbref import DBRef
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _DatabaseAggregationCommand
from pymongo.change_stream import DatabaseChangeStream
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.errors import CollectionInvalid, InvalidName, InvalidOperation
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
def validate_collection(self, name_or_collection: Union[str, Collection[_DocumentTypeArg]], scandata: bool=False, full: bool=False, session: Optional[ClientSession]=None, background: Optional[bool]=None, comment: Optional[Any]=None) -> dict[str, Any]:
    """Validate a collection.

        Returns a dict of validation info. Raises CollectionInvalid if
        validation fails.

        See also the MongoDB documentation on the `validate command`_.

        :Parameters:
          - `name_or_collection`: A Collection object or the name of a
            collection to validate.
          - `scandata`: Do extra checks beyond checking the overall
            structure of the collection.
          - `full`: Have the server do a more thorough scan of the
            collection. Use with `scandata` for a thorough scan
            of the structure of the collection and the individual
            documents.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `background` (optional): A boolean flag that determines whether
            the command runs in the background. Requires MongoDB 4.4+.
          - `comment` (optional): A user-provided comment to attach to this
            command.

        .. versionchanged:: 4.1
           Added ``comment`` parameter.

        .. versionchanged:: 3.11
           Added ``background`` parameter.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. _validate command: https://mongodb.com/docs/manual/reference/command/validate/
        """
    name = name_or_collection
    if isinstance(name, Collection):
        name = name.name
    if not isinstance(name, str):
        raise TypeError('name_or_collection must be an instance of str or Collection')
    cmd = SON([('validate', name), ('scandata', scandata), ('full', full)])
    if comment is not None:
        cmd['comment'] = comment
    if background is not None:
        cmd['background'] = background
    result = self.command(cmd, session=session)
    valid = True
    if 'result' in result:
        info = result['result']
        if info.find('exception') != -1 or info.find('corrupt') != -1:
            raise CollectionInvalid(f'{name} invalid: {info}')
    elif 'raw' in result:
        for _, res in result['raw'].items():
            if 'result' in res:
                info = res['result']
                if info.find('exception') != -1 or info.find('corrupt') != -1:
                    raise CollectionInvalid(f'{name} invalid: {info}')
            elif not res.get('valid', False):
                valid = False
                break
    elif not result.get('valid', False):
        valid = False
    if not valid:
        raise CollectionInvalid(f'{name} invalid: {result!r}')
    return result