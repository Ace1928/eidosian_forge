import codecs
import concurrent.futures
import glob
import os
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import find_command, is_ignored_path, popen_wrapper

        Locations is a list of tuples: [(directory, file), ...]
        