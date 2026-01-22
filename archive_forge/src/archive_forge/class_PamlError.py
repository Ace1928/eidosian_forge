import os
import subprocess
class PamlError(EnvironmentError):
    """paml has failed.

    Run with verbose=True to view the error message.
    """