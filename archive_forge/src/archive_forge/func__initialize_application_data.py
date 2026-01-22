import sys
import os
from os import path
from contextlib import contextmanager
def _initialize_application_data(self, create=True):
    """
        Initializes the (default) application data directory.

        """
    if sys.platform == 'win32':
        environment_variable = 'APPDATA'
        directory_name = self.company
    else:
        environment_variable = 'HOME'
        directory_name = '.' + self.company.lower()
    parent_directory = os.environ.get(environment_variable, None)
    if parent_directory is None or parent_directory == '/root':
        import tempfile
        from warnings import warn
        parent_directory = tempfile.gettempdir()
        user = os.environ.get('USER', None)
        if user is not None:
            directory_name += '_%s' % user
        warn('Environment variable "%s" not set, setting home directory to %s' % (environment_variable, parent_directory))
    application_data = os.path.join(parent_directory, directory_name)
    if create:
        if os.path.exists(application_data):
            if not os.path.isdir(application_data):
                raise ValueError('File "%s" already exists' % application_data)
        else:
            os.makedirs(application_data)
    return application_data