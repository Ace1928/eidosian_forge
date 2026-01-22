from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _check_command_response(response: _DocumentOut, max_wire_version: Optional[int], allowable_errors: Optional[Container[Union[int, str]]]=None, parse_write_concern_error: bool=False) -> None:
    """Check the response to a command for errors."""
    if 'ok' not in response:
        raise OperationFailure(response.get('$err'), response.get('code'), response, max_wire_version)
    if parse_write_concern_error and 'writeConcernError' in response:
        _error = response['writeConcernError']
        _labels = response.get('errorLabels')
        if _labels:
            _error.update({'errorLabels': _labels})
        _raise_write_concern_error(_error)
    if response['ok']:
        return
    details = response
    if 'raw' in response:
        for shard in response['raw'].values():
            if shard.get('errmsg') and (not shard.get('ok')):
                details = shard
                break
    errmsg = details['errmsg']
    code = details.get('code')
    if allowable_errors:
        if code is not None:
            if code in allowable_errors:
                return
        elif errmsg in allowable_errors:
            return
    if code is not None:
        if code in _NOT_PRIMARY_CODES:
            raise NotPrimaryError(errmsg, response)
    elif HelloCompat.LEGACY_ERROR in errmsg or 'node is recovering' in errmsg:
        raise NotPrimaryError(errmsg, response)
    if code in (11000, 11001, 12582):
        raise DuplicateKeyError(errmsg, code, response, max_wire_version)
    elif code == 50:
        raise ExecutionTimeout(errmsg, code, response, max_wire_version)
    elif code == 43:
        raise CursorNotFound(errmsg, code, response, max_wire_version)
    raise OperationFailure(errmsg, code, response, max_wire_version)