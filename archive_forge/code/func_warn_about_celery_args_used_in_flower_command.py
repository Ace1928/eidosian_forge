import os
import sys
import atexit
import signal
import logging
from pprint import pformat
from logging import NullHandler
import click
from tornado.options import options
from tornado.options import parse_command_line, parse_config_file
from tornado.log import enable_pretty_logging
from celery.bin.base import CeleryCommand
from .app import Flower
from .urls import settings
from .utils import abs_path, prepend_url, strtobool
from .options import DEFAULT_CONFIG_FILE, default_options
from .views.auth import validate_auth_option
def warn_about_celery_args_used_in_flower_command(ctx, flower_args):
    celery_options = [option for param in ctx.parent.command.params for option in param.opts]
    incorrectly_used_args = []
    for arg in flower_args:
        arg_name, _, _ = arg.partition('=')
        if arg_name in celery_options:
            incorrectly_used_args.append(arg_name)
    if incorrectly_used_args:
        logger.warning('You have incorrectly specified the following celery arguments after flower command: %s. Please specify them after celery command instead following this template: celery [celery args] flower [flower args].', incorrectly_used_args)