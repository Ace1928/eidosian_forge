import inspect
import os
from argparse import Namespace
from ast import literal_eval
from contextlib import suppress
from functools import wraps
from typing import Any, Callable, Type, TypeVar, cast
def _parse_env_variables(cls: Type, template: str='PL_%(cls_name)s_%(cls_argument)s') -> Namespace:
    """Parse environment arguments if they are defined.

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> _parse_env_variables(Trainer)
        Namespace()
        >>> import os
        >>> os.environ["PL_TRAINER_DEVICES"] = '42'
        >>> os.environ["PL_TRAINER_BLABLABLA"] = '1.23'
        >>> _parse_env_variables(Trainer)
        Namespace(devices=42)
        >>> del os.environ["PL_TRAINER_DEVICES"]

    """
    env_args = {}
    for arg_name in inspect.signature(cls).parameters:
        env = template % {'cls_name': cls.__name__.upper(), 'cls_argument': arg_name.upper()}
        val = os.environ.get(env)
        if not (val is None or val == ''):
            with suppress(Exception):
                val = literal_eval(val)
            env_args[arg_name] = val
    return Namespace(**env_args)