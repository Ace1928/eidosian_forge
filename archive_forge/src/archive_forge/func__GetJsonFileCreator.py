import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetJsonFileCreator(name, json_object):
    """Creates a creator function for an extended source context file.

  Args:
    name: (String) The name of the file to generate.
    json_object: Any object compatible with json.dump.
  Returns:
    (callable()) A creator function that will create the file and return a
    cleanup function that will delete the file.
  """
    if os.path.exists(name):
        logging.warn('%s already exists. It will not be updated.', name)
        return lambda: lambda: None

    def Cleanup():
        os.remove(name)

    def Generate():
        try:
            with open(name, 'w') as f:
                json.dump(json_object, f)
        except IOError as e:
            logging.warn('Could not generate [%s]: %s', name, e)
        return Cleanup
    return Generate