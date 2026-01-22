import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script

    Validate the cmdline args passed into the script.

    :param opt: The arguments of te parser

    :return: Returns extension of output file. None if no output file
    