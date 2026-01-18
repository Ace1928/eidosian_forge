from __future__ import annotations
import functools
import json
import sys
import typing
import click
import httpcore
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table
from ._client import Client
from ._exceptions import RequestError
from ._models import Response
from ._status_codes import codes
def validate_auth(ctx: click.Context, param: click.Option | click.Parameter, value: typing.Any) -> typing.Any:
    if value == (None, None):
        return None
    username, password = value
    if password == '-':
        password = click.prompt('Password', hide_input=True)
    return (username, password)