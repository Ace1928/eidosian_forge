import sys
import os
from os import path
from contextlib import contextmanager
def _initialize_user_data(self):
    """
        Initializes the (default) user data directory.

        """
    parent_directory = os.path.expanduser('~')
    directory_name = self.company
    if sys.platform == 'win32':
        try:
            from win32com.shell import shell, shellcon
            MY_DOCS = shellcon.CSIDL_PERSONAL
            parent_directory = shell.SHGetFolderPath(0, MY_DOCS, 0, 0)
        except ImportError:
            desired_dir = os.path.join(parent_directory, 'My Documents')
            if os.path.exists(desired_dir):
                parent_directory = desired_dir
    else:
        directory_name = directory_name.lower()
    usr_dir = os.path.join(parent_directory, directory_name)
    if os.path.exists(usr_dir):
        if not os.path.isdir(usr_dir):
            raise ValueError('File "%s" already exists' % usr_dir)
    else:
        os.makedirs(usr_dir)
    return usr_dir