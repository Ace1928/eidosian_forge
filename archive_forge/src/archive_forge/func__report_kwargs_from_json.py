import dataclasses
from io import StringIO
import os
from pprint import pprint
from typing import Any
from typing import cast
from typing import Dict
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import skip
def _report_kwargs_from_json(reportdict: Dict[str, Any]) -> Dict[str, Any]:
    """Return **kwargs that can be used to construct a TestReport or
    CollectReport instance.

    This was originally the serialize_report() function from xdist (ca03269).
    """

    def deserialize_repr_entry(entry_data):
        data = entry_data['data']
        entry_type = entry_data['type']
        if entry_type == 'ReprEntry':
            reprfuncargs = None
            reprfileloc = None
            reprlocals = None
            if data['reprfuncargs']:
                reprfuncargs = ReprFuncArgs(**data['reprfuncargs'])
            if data['reprfileloc']:
                reprfileloc = ReprFileLocation(**data['reprfileloc'])
            if data['reprlocals']:
                reprlocals = ReprLocals(data['reprlocals']['lines'])
            reprentry: Union[ReprEntry, ReprEntryNative] = ReprEntry(lines=data['lines'], reprfuncargs=reprfuncargs, reprlocals=reprlocals, reprfileloc=reprfileloc, style=data['style'])
        elif entry_type == 'ReprEntryNative':
            reprentry = ReprEntryNative(data['lines'])
        else:
            _report_unserialization_failure(entry_type, TestReport, reportdict)
        return reprentry

    def deserialize_repr_traceback(repr_traceback_dict):
        repr_traceback_dict['reprentries'] = [deserialize_repr_entry(x) for x in repr_traceback_dict['reprentries']]
        return ReprTraceback(**repr_traceback_dict)

    def deserialize_repr_crash(repr_crash_dict: Optional[Dict[str, Any]]):
        if repr_crash_dict is not None:
            return ReprFileLocation(**repr_crash_dict)
        else:
            return None
    if reportdict['longrepr'] and 'reprcrash' in reportdict['longrepr'] and ('reprtraceback' in reportdict['longrepr']):
        reprtraceback = deserialize_repr_traceback(reportdict['longrepr']['reprtraceback'])
        reprcrash = deserialize_repr_crash(reportdict['longrepr']['reprcrash'])
        if reportdict['longrepr']['chain']:
            chain = []
            for repr_traceback_data, repr_crash_data, description in reportdict['longrepr']['chain']:
                chain.append((deserialize_repr_traceback(repr_traceback_data), deserialize_repr_crash(repr_crash_data), description))
            exception_info: Union[ExceptionChainRepr, ReprExceptionInfo] = ExceptionChainRepr(chain)
        else:
            exception_info = ReprExceptionInfo(reprtraceback=reprtraceback, reprcrash=reprcrash)
        for section in reportdict['longrepr']['sections']:
            exception_info.addsection(*section)
        reportdict['longrepr'] = exception_info
    return reportdict