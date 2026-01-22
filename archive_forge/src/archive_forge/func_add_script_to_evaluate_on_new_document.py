from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def add_script_to_evaluate_on_new_document(source: str, world_name: typing.Optional[str]=None, include_command_line_api: typing.Optional[bool]=None, run_immediately: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ScriptIdentifier]:
    """
    Evaluates given script in every frame upon creation (before loading frame's scripts).

    :param source:
    :param world_name: **(EXPERIMENTAL)** *(Optional)* If specified, creates an isolated world with the given name and evaluates given script in it. This world name will be used as the ExecutionContextDescription::name when the corresponding event is emitted.
    :param include_command_line_api: **(EXPERIMENTAL)** *(Optional)* Specifies whether command line API should be available to the script, defaults to false.
    :param run_immediately: **(EXPERIMENTAL)** *(Optional)* If true, runs the script immediately on existing execution contexts or worlds. Default: false.
    :returns: Identifier of the added script.
    """
    params: T_JSON_DICT = dict()
    params['source'] = source
    if world_name is not None:
        params['worldName'] = world_name
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if run_immediately is not None:
        params['runImmediately'] = run_immediately
    cmd_dict: T_JSON_DICT = {'method': 'Page.addScriptToEvaluateOnNewDocument', 'params': params}
    json = (yield cmd_dict)
    return ScriptIdentifier.from_json(json['identifier'])