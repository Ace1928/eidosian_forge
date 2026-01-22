import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
def _nodes_from_argv(self, argv, cmd=None):
    cmd = cmd if cmd is not None else self.cmd
    p = self.OptionParser(argv)
    p.parse()
    return (p, self.MultiParser(cmd=cmd).parse(p))