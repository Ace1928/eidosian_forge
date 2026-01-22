from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def _call_process(self, method: str, *args: Any, **kwargs: Any) -> Union[str, bytes, Tuple[int, Union[str, bytes], str], 'Git.AutoInterrupt']:
    """Run the given git command with the specified arguments and return
        the result as a string.

        :param method:
            The command. Contained ``_`` characters will be converted to dashes,
            such as in ``ls_files`` to call ``ls-files``.

        :param args:
            The list of arguments. If None is included, it will be pruned.
            This allows your commands to call git more conveniently as None
            is realized as non-existent.

        :param kwargs:
            Contains key-values for the following:
            - The :meth:`execute()` kwds, as listed in :var:`execute_kwargs`.
            - "Command options" to be converted by :meth:`transform_kwargs`.
            - The `'insert_kwargs_after'` key which its value must match one of ``*args``.
            It also contains any command options, to be appended after the matched arg.

        Examples::

            git.rev_list('master', max_count=10, header=True)

        turns into::

           git rev-list max-count 10 --header master

        :return: Same as :meth:`execute`.
            If no args are given, used :meth:`execute`'s default (especially
            ``as_process = False``, ``stdout_as_string = True``) and return str.
        """
    exec_kwargs = {k: v for k, v in kwargs.items() if k in execute_kwargs}
    opts_kwargs = {k: v for k, v in kwargs.items() if k not in execute_kwargs}
    insert_after_this_arg = opts_kwargs.pop('insert_kwargs_after', None)
    opt_args = self.transform_kwargs(**opts_kwargs)
    ext_args = self._unpack_args([a for a in args if a is not None])
    if insert_after_this_arg is None:
        args_list = opt_args + ext_args
    else:
        try:
            index = ext_args.index(insert_after_this_arg)
        except ValueError as err:
            raise ValueError("Couldn't find argument '%s' in args %s to insert cmd options after" % (insert_after_this_arg, str(ext_args))) from err
        args_list = ext_args[:index + 1] + opt_args + ext_args[index + 1:]
    call = [self.GIT_PYTHON_GIT_EXECUTABLE]
    call.extend(self._persistent_git_options)
    call.extend(self._git_options)
    self._git_options = ()
    call.append(dashify(method))
    call.extend(args_list)
    return self.execute(call, **exec_kwargs)