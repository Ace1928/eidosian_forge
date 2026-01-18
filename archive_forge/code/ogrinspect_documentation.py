import argparse
from django.contrib.gis import gdal
from django.core.management.base import BaseCommand, CommandError
from django.utils.inspect import get_func_args

    Custom argparse action for `ogrinspect` keywords that require
    a string list. If the string is 'True'/'true' then the option
    value will be a boolean instead.
    