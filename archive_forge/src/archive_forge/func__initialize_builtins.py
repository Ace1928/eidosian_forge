import logging
import os
def _initialize_builtins():
    """Scan the immediate subdirectories of the builtins module.

  Encountered subdirectories with an app.yaml file are added to
  AVAILABLE_BUILTINS.
  """
    if os.path.isdir(_handler_dir):
        for filename in os.listdir(_handler_dir):
            if os.path.isfile(_get_yaml_path(filename, '')):
                _available_builtins.append(filename)