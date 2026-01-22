import dataclasses
import json
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints
import yaml
def HfArg(*, aliases: Union[str, List[str]]=None, help: str=None, default: Any=dataclasses.MISSING, default_factory: Callable[[], Any]=dataclasses.MISSING, metadata: dict=None, **kwargs) -> dataclasses.Field:
    """Argument helper enabling a concise syntax to create dataclass fields for parsing with `HfArgumentParser`.

    Example comparing the use of `HfArg` and `dataclasses.field`:
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```

    Args:
        aliases (Union[str, List[str]], optional):
            Single string or list of strings of aliases to pass on to argparse, e.g. `aliases=["--example", "-e"]`.
            Defaults to None.
        help (str, optional): Help string to pass on to argparse that can be displayed with --help. Defaults to None.
        default (Any, optional):
            Default value for the argument. If not default or default_factory is specified, the argument is required.
            Defaults to dataclasses.MISSING.
        default_factory (Callable[[], Any], optional):
            The default_factory is a 0-argument function called to initialize a field's value. It is useful to provide
            default values for mutable types, e.g. lists: `default_factory=list`. Mutually exclusive with `default=`.
            Defaults to dataclasses.MISSING.
        metadata (dict, optional): Further metadata to pass on to `dataclasses.field`. Defaults to None.

    Returns:
        Field: A `dataclasses.Field` with the desired properties.
    """
    if metadata is None:
        metadata = {}
    if aliases is not None:
        metadata['aliases'] = aliases
    if help is not None:
        metadata['help'] = help
    return dataclasses.field(metadata=metadata, default=default, default_factory=default_factory, **kwargs)