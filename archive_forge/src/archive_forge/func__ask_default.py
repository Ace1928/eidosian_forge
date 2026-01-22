import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
def _ask_default(self, default=''):
    """
        Prompt for a default value.

        The ``default`` argument allows providing a custom default value (as a
        string) which will be shown to the user and used as the return value
        if the user doesn't provide any other input.
        """
    self.prompt_output.write('Please enter the default value as valid Python.')
    if default:
        self.prompt_output.write(f"Accept the default '{default}' by pressing 'Enter' or provide another value.")
    self.prompt_output.write('The datetime and django.utils.timezone modules are available, so it is possible to provide e.g. timezone.now as a value.')
    self.prompt_output.write("Type 'exit' to exit this prompt")
    while True:
        if default:
            prompt = '[default: {}] >>> '.format(default)
        else:
            prompt = '>>> '
        self.prompt_output.write(prompt, ending='')
        code = input()
        if not code and default:
            code = default
        if not code:
            self.prompt_output.write("Please enter some code, or 'exit' (without quotes) to exit.")
        elif code == 'exit':
            sys.exit(1)
        else:
            try:
                return eval(code, {}, {'datetime': datetime, 'timezone': timezone})
            except (SyntaxError, NameError) as e:
                self.prompt_output.write('Invalid input: %s' % e)