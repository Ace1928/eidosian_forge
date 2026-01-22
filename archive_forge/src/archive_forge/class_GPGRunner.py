from __future__ import (absolute_import, division, print_function)
import abc
import os
from ansible.module_utils import six
@six.add_metaclass(abc.ABCMeta)
class GPGRunner(object):

    @abc.abstractmethod
    def run_command(self, command, check_rc=True, data=None):
        """
        Run ``[gpg] + command`` and return ``(rc, stdout, stderr)``.

        If ``data`` is not ``None``, it will be provided as stdin.
        The code assumes it is a bytes string.

        Returned stdout and stderr are native Python strings.
        Pass ``check_rc=False`` to allow return codes != 0.

        Raises a ``GPGError`` in case of errors.
        """
        pass