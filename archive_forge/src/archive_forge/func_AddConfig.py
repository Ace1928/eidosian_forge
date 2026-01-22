import os
import re
import socket  # for gethostname
import gyp.easy_xml as easy_xml
def AddConfig(self, name):
    """Adds a configuration to the project.

    Args:
      name: Configuration name.
    """
    self.configurations[name] = ['Configuration', {'Name': name}]