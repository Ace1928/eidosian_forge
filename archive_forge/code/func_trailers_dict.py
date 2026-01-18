import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@property
def trailers_dict(self) -> Dict[str, List[str]]:
    """Get the trailers of the message as a dictionary.

        Git messages can contain trailer information that are similar to RFC 822
        e-mail headers (see: https://git-scm.com/docs/git-interpret-trailers).

        This functions calls ``git interpret-trailers --parse`` onto the message
        to extract the trailer information. The key value pairs are stripped of
        leading and trailing whitespaces before they get saved into a dictionary.

        Valid message with trailer::

            Subject line

            some body information

            another information

            key1: value1.1
            key1: value1.2
            key2 :    value 2 with inner spaces


        Returned dictionary will look like this::

            {
                "key1": ["value1.1", "value1.2"],
                "key2": ["value 2 with inner spaces"],
            }


        :return:
            Dictionary containing whitespace stripped trailer information.
            Mapping trailer keys to a list of their corresponding values.
        """
    d = defaultdict(list)
    for key, val in self.trailers_list:
        d[key].append(val)
    return dict(d)